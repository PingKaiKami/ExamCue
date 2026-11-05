import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from examcue.domain import MediaItem, TranscriptSegment
from examcue.pipeline import ExamCuePipeline, PipelineCancelled, PipelineConfig


class PipelineControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp_directory.cleanup)

    def _pipeline(
        self, cancel_event: threading.Event, resume_event: threading.Event
    ) -> ExamCuePipeline:
        config = PipelineConfig("test", Path(self.temp_directory.name), [])
        return ExamCuePipeline(config, cancel_event=cancel_event, resume_event=resume_event)

    def test_paused_pipeline_waits_until_resumed(self):
        cancel_event = threading.Event()
        resume_event = threading.Event()
        pipeline = self._pipeline(cancel_event, resume_event)
        resume_event.clear()
        completed = threading.Event()

        worker = threading.Thread(target=lambda: (pipeline._check_control(), completed.set()))
        worker.start()
        time.sleep(0.2)
        self.assertFalse(completed.is_set())

        resume_event.set()
        worker.join(timeout=1)
        self.assertTrue(completed.is_set())

    def test_cancel_releases_a_paused_pipeline(self):
        cancel_event = threading.Event()
        resume_event = threading.Event()
        pipeline = self._pipeline(cancel_event, resume_event)
        resume_event.clear()
        errors: list[BaseException] = []

        def run_control() -> None:
            try:
                pipeline._check_control()
            except PipelineCancelled as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_control)
        worker.start()
        cancel_event.set()
        worker.join(timeout=1)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PipelineCancelled)

    def test_next_source_prepares_while_current_source_transcribes(self):
        second_prepare_started = threading.Event()
        allow_second_prepare_to_finish = threading.Event()
        root = Path(self.temp_directory.name)

        class FakePreparer:
            def __init__(self, *_args, **_kwargs):
                pass

            def prepare(self, source, _progress):
                item_id = source.rsplit("/", 1)[-1]
                if item_id == "two":
                    second_prepare_started.set()
                    allow_second_prepare_to_finish.wait(timeout=2)
                return [
                    MediaItem(
                        source=source,
                        item_id=item_id,
                        title=item_id,
                        source_type="youtube",
                        audio_path=root / f"{item_id}.webm",
                    )
                ]

            def pop_warnings(self):
                return []

        class FakeTranscriber:
            last_method = "whisper"
            device = "cpu"
            compute_type = "int8"

            def __init__(self, *_args, **_kwargs):
                self.profile = {"beam_size": 3, "batch_size": 4}

            def transcribe(self, item, _progress, _control):
                if item.item_id == "one":
                    if not second_prepare_started.wait(timeout=2):
                        raise AssertionError("next download did not overlap transcription")
                    allow_second_prepare_to_finish.set()
                return [TranscriptSegment(0, 1, item.title)], "zh"

        config = PipelineConfig(
            "overlap",
            root,
            ["https://youtube/one", "https://youtube/two"],
        )
        analyzer = Mock(warnings=[])
        analyzer.estimate_work.return_value = 1
        analyzer.client = Mock(
            status=None,
            analysis_request_count=0,
            analysis_request_seconds=0.0,
        )
        analyzer.analysis_chunk_count = 0
        analyzer.analysis_chunk_cache_hits = 0
        analyzer.llm_failure_count = 0
        analyzer.fallback_chunk_count = 0
        analyzer.quality_filtered_segment_count = 0
        analyzer.quality_rejected_cue_count = 0
        analyzer.empty_direct_response_count = 0
        messages: list[str] = []
        with (
            patch("examcue.pipeline.MediaPreparer", FakePreparer),
            patch("examcue.pipeline.Transcriber", FakeTranscriber),
            patch("examcue.pipeline.CueAnalyzer", return_value=analyzer),
            patch("examcue.pipeline.write_report"),
            patch.object(ExamCuePipeline, "_load_analysis_cache", return_value=[]),
        ):
            report = ExamCuePipeline(config, progress=messages.append).run()

        self.assertEqual([source["item_id"] for source in report.sources], ["one", "two"])
        self.assertIn("整體分析工作進度 1/2", messages)
        self.assertIn("整體分析工作進度 2/2", messages)
        metrics = json.loads((root / "run_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["media_item_count"], 2)
        self.assertEqual(metrics["total_media_duration_seconds"], 2.0)
        self.assertEqual(metrics["llm"]["analysis_request_count"], 0)
        self.assertEqual(metrics["llm"]["quality_filtered_segment_count"], 0)
        self.assertEqual(metrics["llm"]["quality_rejected_cue_count"], 0)
        self.assertEqual(metrics["llm"]["empty_direct_response_count"], 0)
        self.assertEqual(metrics["sources"][0]["transcription_device"], "cpu")
        self.assertEqual(metrics["sources"][0]["transcription_compute_type"], "int8")
        self.assertEqual(metrics["sources"][0]["transcription_batch_size"], 4)
        self.assertTrue((root / "run_history.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
