from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from examcue.domain import MediaItem
from examcue.media import MediaPreparationError, MediaPreparer, expand_youtube_urls


class _FakeYoutubeDL:
    def __init__(self, _options):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def extract_info(self, _url, download=False):
        if download:
            raise AssertionError("playlist extraction must not download")
        return {
            "entries": [
                {"id": "ok-1", "title": "第一支"},
                {"id": "bad-2", "title": "第二支"},
                {"id": "ok-3", "title": "第三支"},
            ]
        }


class MediaPreparerTests(unittest.TestCase):
    def _preparer(self, root: Path) -> MediaPreparer:
        return MediaPreparer(root)

    def test_playlist_keeps_successful_items_when_one_video_fails(self):
        fake_yt_dlp = types.SimpleNamespace(YoutubeDL=_FakeYoutubeDL)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            preparer = self._preparer(Path(directory))

            def download(url, _progress):
                video_id = url.rsplit("=", 1)[-1]
                if video_id == "bad-2":
                    raise MediaPreparationError("HTTP Error 403: Forbidden")
                return MediaItem(
                    source=url,
                    item_id=video_id,
                    title=video_id,
                    source_type="youtube",
                    audio_path=Path(directory) / f"{video_id}.wav",
                )

            with (
                patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
                patch.object(preparer, "_download_youtube_item", side_effect=download),
            ):
                items = preparer._prepare_youtube("https://youtube/playlist", Mock())

        self.assertEqual([item.item_id for item in items], ["ok-1", "ok-3"])
        self.assertEqual(len(preparer.warnings), 1)
        self.assertIn("第 2 項", preparer.warnings[0])
        self.assertIn("403", preparer.warnings[0])

    def test_local_sidecar_subtitle_skips_audio_extraction(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            video = root / "lecture.mp4"
            subtitle = root / "lecture.srt"
            video.touch()
            subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n考試重點\n",
                encoding="utf-8",
            )
            preparer = MediaPreparer(root / "media", transcription_source="auto")

            item = preparer._prepare_local(video, Mock())

        self.assertEqual(item.subtitle_path, subtitle)
        self.assertEqual(item.subtitle_kind, "sidecar")
        self.assertIsNone(item.audio_path)

    def test_local_video_is_passed_directly_to_whisper_without_conversion(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            video = root / "lecture.mp4"
            video.touch()
            preparer = MediaPreparer(root / "media", transcription_source="whisper")
            progress = Mock()

            item = preparer._prepare_local(video, progress)

        self.assertEqual(item.audio_path, video.resolve())
        self.assertIn("直接讀取原始影音", progress.call_args.args[0])

    def test_playlist_is_expanded_into_individual_video_urls(self):
        fake_yt_dlp = types.SimpleNamespace(
            YoutubeDL=_FakeYoutubeDL,
            utils=types.SimpleNamespace(DownloadError=RuntimeError),
        )
        with patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
            urls = expand_youtube_urls("https://youtube/playlist?list=course")

        self.assertEqual(
            urls,
            [
                "https://www.youtube.com/watch?v=ok-1",
                "https://www.youtube.com/watch?v=bad-2",
                "https://www.youtube.com/watch?v=ok-3",
            ],
        )

    def test_youtube_download_retries_with_fresh_extraction(self):
        class DownloadError(Exception):
            pass

        fake_yt_dlp = types.SimpleNamespace(
            utils=types.SimpleNamespace(DownloadError=DownloadError)
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            preparer = self._preparer(Path(directory))
            item = MediaItem(
                source="https://youtube/watch?v=abc",
                item_id="abc",
                title="影片",
                source_type="youtube",
                audio_path=Path(directory) / "audio.wav",
            )
            operation = Mock(side_effect=[DownloadError("403"), item])
            progress = Mock()
            with (
                patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
                patch.object(preparer, "_download_youtube_item_once", operation),
            ):
                result = preparer._download_youtube_item(item.source, progress)

        self.assertIs(result, item)
        self.assertEqual(operation.call_count, 2)
        self.assertIn("重試 (2/3)", progress.call_args.args[0])

    def test_single_video_reuses_initial_metadata_extraction(self):
        info = {
            "id": "single",
            "title": "Single video",
            "webpage_url": "https://youtube/watch?v=single",
        }

        class InfoYoutubeDL:
            def __init__(self, _options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def extract_info(self, _url, download=False):
                if download:
                    raise AssertionError("metadata lookup must not download")
                return info

        fake_yt_dlp = types.SimpleNamespace(YoutubeDL=InfoYoutubeDL)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            preparer = self._preparer(Path(directory))
            item = MediaItem(
                source=info["webpage_url"],
                item_id="single",
                title="Single video",
                source_type="youtube",
            )
            with (
                patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
                patch.object(preparer, "_download_youtube_item", return_value=item) as download,
            ):
                result = preparer._prepare_youtube(info["webpage_url"], Mock())

        self.assertEqual(result, [item])
        self.assertIs(download.call_args.kwargs["initial_info"], info)

    def test_automatic_subtitle_skips_audio_download(self):
        class InfoYoutubeDL:
            def __init__(self, _options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def extract_info(self, _url, download=False):
                self.assert_not_downloading(download)
                return {
                    "id": "captioned",
                    "title": "Captioned lecture",
                    "webpage_url": "https://youtube/watch?v=captioned",
                }

            @staticmethod
            def assert_not_downloading(download):
                if download:
                    raise AssertionError("must only extract metadata")

            def download(self, _urls):
                raise AssertionError("audio download should be skipped")

        fake_yt_dlp = types.SimpleNamespace(YoutubeDL=InfoYoutubeDL)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            subtitle = root / "automatic.zh-Hant.vtt"
            subtitle.write_text("WEBVTT", encoding="utf-8")
            preparer = self._preparer(root)
            with (
                patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
                patch.object(
                    preparer,
                    "_download_preferred_subtitle",
                    return_value=(subtitle, "automatic"),
                ),
            ):
                item = preparer._download_youtube_item_once(
                    "https://youtube/watch?v=captioned", Mock()
                )

        self.assertEqual(item.subtitle_path, subtitle)
        self.assertEqual(item.subtitle_kind, "automatic")
        self.assertIsNone(item.audio_path)

    def test_youtube_metadata_cache_skips_network_on_second_prepare(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            item_dir = root / "youtube-cached123"
            item_dir.mkdir()
            audio = item_dir / "source.webm"
            audio.write_bytes(b"cached audio")
            item = MediaItem(
                source="https://youtu.be/cached123?si=share",
                item_id="youtube-cached123",
                title="Cached lecture",
                source_type="youtube",
                audio_path=audio,
                webpage_url="https://www.youtube.com/watch?v=cached123",
            )
            MediaPreparer._save_youtube_metadata(item_dir, item)
            preparer = MediaPreparer(root, transcription_source="auto")

            with patch.object(
                preparer,
                "_prepare_youtube",
                side_effect=AssertionError("cache hit must not access YouTube"),
            ):
                result = preparer.prepare(item.source, Mock())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "Cached lecture")
        self.assertEqual(result[0].audio_path, audio)

    def test_youtube_cache_prefers_subtitle_in_auto_mode(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            item_dir = root / "youtube-caption123"
            item_dir.mkdir()
            audio = item_dir / "source.webm"
            subtitle = item_dir / "automatic.zh-Hant.vtt"
            audio.write_bytes(b"cached audio")
            subtitle.write_text("WEBVTT", encoding="utf-8")
            item = MediaItem(
                source="https://www.youtube.com/watch?v=caption123",
                item_id="youtube-caption123",
                title="Captioned cached lecture",
                source_type="youtube",
                audio_path=audio,
                subtitle_path=subtitle,
                subtitle_kind="automatic",
            )
            MediaPreparer._save_youtube_metadata(item_dir, item)

            cached = MediaPreparer(root, transcription_source="auto").prepare(item.source, Mock())[
                0
            ]

        self.assertEqual(cached.subtitle_path, subtitle)
        self.assertEqual(cached.subtitle_kind, "automatic")
        self.assertIsNone(cached.audio_path)

    def test_whisper_mode_requires_cached_audio(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            item_dir = root / "youtube-captiononly"
            item_dir.mkdir()
            subtitle = item_dir / "automatic.zh-Hant.vtt"
            subtitle.write_text("WEBVTT", encoding="utf-8")
            item = MediaItem(
                source="https://www.youtube.com/watch?v=captiononly",
                item_id="youtube-captiononly",
                title="Subtitle only",
                source_type="youtube",
                subtitle_path=subtitle,
                subtitle_kind="automatic",
            )
            MediaPreparer._save_youtube_metadata(item_dir, item)
            preparer = MediaPreparer(root, transcription_source="whisper")

            self.assertIsNone(preparer._load_cached_youtube_item(item.source))

    def test_youtube_video_id_parses_watch_and_short_urls(self):
        self.assertEqual(
            MediaPreparer._youtube_video_id(
                "https://www.youtube.com/watch?v=mC3MOlqY1Tg&list=course"
            ),
            "mC3MOlqY1Tg",
        )

    def test_youtube_audio_prefers_transcription_sized_bitrate(self):
        options_seen: list[dict] = []

        class DownloadYoutubeDL:
            def __init__(self, options):
                self.options = options
                options_seen.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def extract_info(self, _url, download=False):
                if download:
                    raise AssertionError("metadata lookup must not download")
                return {
                    "id": "audioonly",
                    "title": "Audio only",
                    "webpage_url": "https://www.youtube.com/watch?v=audioonly",
                }

            def download(self, _urls):
                output = Path(self.options["outtmpl"].replace("%(ext)s", "webm"))
                output.write_bytes(b"audio")

        fake_yt_dlp = types.SimpleNamespace(YoutubeDL=DownloadYoutubeDL)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            preparer = MediaPreparer(Path(directory), transcription_source="auto")
            with patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
                item = preparer._download_youtube_item_once(
                    "https://www.youtube.com/watch?v=audioonly", Mock()
                )

        download_options = next(options for options in options_seen if "format" in options)
        self.assertEqual(download_options["format"], "bestaudio[abr<=128]/bestaudio/best")
        self.assertEqual(item.audio_path.suffix, ".webm")

    def test_metadata_cache_failure_does_not_discard_downloaded_media(self):
        fake_yt_dlp = types.SimpleNamespace()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            item_dir = root / "youtube-alreadydownloaded"
            item_dir.mkdir()
            audio = item_dir / "source.webm"
            audio.write_bytes(b"audio")
            preparer = MediaPreparer(root, transcription_source="whisper")
            info = {
                "id": "alreadydownloaded",
                "title": "Already downloaded",
                "webpage_url": "https://www.youtube.com/watch?v=alreadydownloaded",
            }
            with (
                patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
                patch.object(
                    preparer,
                    "_save_youtube_metadata",
                    side_effect=OSError("disk busy"),
                ),
            ):
                item = preparer._download_youtube_item_once(info["webpage_url"], Mock(), info=info)

        self.assertEqual(item.audio_path, audio)
        self.assertTrue(any("快取寫入失敗" in warning for warning in preparer.warnings))
        self.assertEqual(
            MediaPreparer._youtube_video_id("https://youtu.be/mC3MOlqY1Tg?si=share"),
            "mC3MOlqY1Tg",
        )
        self.assertEqual(
            MediaPreparer._youtube_video_id("https://www.youtube.com/shorts/mC3MOlqY1Tg"),
            "mC3MOlqY1Tg",
        )


if __name__ == "__main__":
    unittest.main()
