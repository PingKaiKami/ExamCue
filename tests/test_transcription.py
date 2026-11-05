import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from examcue.domain import MediaItem, TranscriptSegment
from examcue.transcription import (
    Transcriber,
    TranscriptStore,
    _is_cuda_runtime_error,
    cuda_diagnostics,
    parse_subtitle,
)


class SubtitleTests(unittest.TestCase):
    def test_reports_timestamp_based_transcription_progress(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            transcriber = Transcriber(TranscriptStore(Path(directory)), "tiny", mode="accurate")
            fake_model = Mock()
            fake_model.transcribe.return_value = (
                [
                    SimpleNamespace(start=0.0, end=25.0, text="first"),
                    SimpleNamespace(start=25.0, end=100.0, text="second"),
                ],
                SimpleNamespace(language="en", duration=100.0),
            )
            transcriber._get_model = Mock(return_value=fake_model)
            messages: list[str] = []

            segments, language = transcriber._transcribe_audio(
                Path(directory) / "audio.wav", progress=messages.append
            )

        self.assertEqual(language, "en")
        self.assertEqual(len(segments), 2)
        self.assertEqual(
            messages,
            ["語音辨識進度 25.00/100.00", "語音辨識進度 100.00/100.00"],
        )

    def test_standard_mode_uses_batched_inference(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            transcriber = Transcriber(TranscriptStore(Path(directory)), "small", mode="standard")
            base_model = Mock()
            batched_model = Mock()
            batched_model.transcribe.return_value = (
                [SimpleNamespace(start=0.0, end=10.0, text="batch")],
                SimpleNamespace(language="en", duration=10.0),
            )
            transcriber._get_model = Mock(return_value=base_model)
            transcriber._get_batched_model = Mock(return_value=batched_model)

            segments, _language = transcriber._transcribe_audio(Path(directory) / "audio.webm")

        self.assertEqual([segment.text for segment in segments], ["batch"])
        self.assertEqual(batched_model.transcribe.call_args.kwargs["batch_size"], 4)
        self.assertEqual(batched_model.transcribe.call_args.kwargs["beam_size"], 3)

    def test_batch_failure_falls_back_to_single_inference(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            transcriber = Transcriber(TranscriptStore(Path(directory)), "small", mode="fast")
            base_model = Mock()
            base_model.transcribe.return_value = (
                [SimpleNamespace(start=0.0, end=5.0, text="fallback")],
                SimpleNamespace(language="zh", duration=5.0),
            )
            batched_model = Mock()
            batched_model.transcribe.side_effect = RuntimeError("out of memory")
            transcriber._get_model = Mock(return_value=base_model)
            transcriber._get_batched_model = Mock(return_value=batched_model)
            messages: list[str] = []

            segments, _language = transcriber._transcribe_audio(
                Path(directory) / "audio.webm", progress=messages.append
            )

        self.assertEqual([segment.text for segment in segments], ["fallback"])
        self.assertIn("批次辨識無法執行，改用單批模式繼續", messages)
        base_model.transcribe.assert_called_once()

    def test_transcript_cache_is_invalidated_when_model_changes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TranscriptStore(Path(directory))
            item = MediaItem("video.mp4", "model-cache", "Model cache", "file")
            expected = [TranscriptSegment(0, 1, "model-specific transcript")]
            store.save(item, expected, "en", "small")

            self.assertEqual(store.load(item, expected_model="small"), expected)
            self.assertIsNone(store.load(item, expected_model="large-v3"))

    def test_transcript_cache_is_invalidated_when_mode_changes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            store = TranscriptStore(Path(directory))
            item = MediaItem("video.mp4", "mode-cache", "Mode cache", "file")
            expected = [TranscriptSegment(0, 1, "mode-specific transcript")]
            store.save(item, expected, "en", "small:fast")

            self.assertEqual(store.load(item, expected_model="small:fast"), expected)
            self.assertIsNone(store.load(item, expected_model="small:standard"))

    def test_parses_srt_and_merges_duplicate_caption(self):
        content = """1
00:00:01,000 --> 00:00:03,000
This will be on the exam.

2
00:00:03,000 --> 00:00:05,000
This will be on the exam.
"""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "lecture.srt"
            path.write_text(content, encoding="utf-8")
            segments = parse_subtitle(path)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start, 1)
        self.assertEqual(segments[0].end, 5)

    def test_cuda_library_error_retries_on_cpu(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            audio.touch()
            transcriber = Transcriber(TranscriptStore(root / "transcripts"), "tiny")
            transcriber._device = "cuda"
            expected = [TranscriptSegment(0, 1, "test")]
            transcriber._transcribe_audio = Mock(
                side_effect=[RuntimeError("Library cublas64_12.dll is not found"), (expected, "en")]
            )
            item = MediaItem(
                source="video.mp4",
                item_id="cuda-fallback",
                title="CUDA fallback",
                source_type="file",
                audio_path=audio,
            )

            segments, language = transcriber.transcribe(item)

        self.assertEqual(segments, expected)
        self.assertEqual(language, "en")
        self.assertEqual(
            transcriber._transcribe_audio.call_args_list[1].kwargs,
            {"force_cpu": True, "progress": ANY, "control": ANY},
        )

    def test_identifies_cuda_runtime_errors(self):
        self.assertTrue(_is_cuda_runtime_error(RuntimeError("cublas64_12.dll missing")))
        self.assertFalse(_is_cuda_runtime_error(RuntimeError("invalid audio format")))

    def test_cuda_diagnostics_explains_missing_windows_libraries(self):
        fake_ctranslate2 = SimpleNamespace(
            __version__="test",
            get_cuda_device_count=lambda: 1,
        )
        fake_smi = SimpleNamespace(
            returncode=0,
            stdout="Test NVIDIA GPU\n",
        )
        with (
            patch.dict(sys.modules, {"ctranslate2": fake_ctranslate2}),
            patch("examcue.transcription.sys.platform", "win32"),
            patch("examcue.transcription.configure_cuda_dll_directories", return_value=[]),
            patch("examcue.transcription.ctypes.WinDLL", side_effect=OSError("missing")),
            patch("examcue.transcription.subprocess.run", return_value=fake_smi),
        ):
            result = cuda_diagnostics()

        self.assertFalse(result["cuda_ready"])
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["gpu_name"], "Test NVIDIA GPU")
        self.assertEqual(
            result["missing_dlls"],
            ["cublas64_12.dll", "cudnn64_9.dll"],
        )


if __name__ == "__main__":
    unittest.main()
