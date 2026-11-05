from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

from .domain import MediaItem
from .util import atomic_write_json, is_url, stable_id

ProgressCallback = Callable[[str], None]


class MediaPreparationError(RuntimeError):
    pass


def expand_youtube_urls(url: str, progress: ProgressCallback = lambda _: None) -> list[str]:
    """Return canonical video URLs, expanding a playlist without downloading media."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise MediaPreparationError("尚未安裝 yt-dlp，無法讀取 YouTube 播放清單。") from exc

    progress(f"正在讀取 YouTube 連結：{url}")
    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "extractor_retries": 3,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except (yt_dlp.utils.DownloadError, OSError) as exc:
        raise MediaPreparationError(f"無法讀取 YouTube 連結：{exc}") from exc

    raw_entries = info.get("entries")
    if raw_entries is None:
        return [str(info.get("webpage_url") or url)]

    video_urls: list[str] = []
    for entry in raw_entries:
        if not entry:
            continue
        video_url = entry.get("webpage_url") or entry.get("url")
        if not video_url or not str(video_url).startswith("http"):
            video_id = entry.get("id") or video_url
            if not video_id:
                continue
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_urls.append(str(video_url))

    video_urls = list(dict.fromkeys(video_urls))
    if not video_urls:
        raise MediaPreparationError("播放清單中找不到可用的影片連結。")
    progress(f"播放清單已展開為 {len(video_urls)} 支影片")
    return video_urls


class MediaPreparer:
    YOUTUBE_DOWNLOAD_ATTEMPTS: ClassVar[int] = 3
    SUPPORTED_LOCAL: ClassVar[set[str]] = {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        ".mp3",
        ".m4a",
        ".wav",
        ".aac",
        ".flac",
        ".ogg",
    }

    def __init__(
        self,
        work_dir: Path,
        project_root: Path | None = None,
        transcription_source: str = "auto",
        capture_video: bool = False,
    ):
        del project_root  # Kept for API compatibility with existing callers.
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.transcription_source = transcription_source
        self.capture_video = capture_video
        self.warnings: list[str] = []

    def pop_warnings(self) -> list[str]:
        warnings, self.warnings = self.warnings, []
        return warnings

    def prepare(self, source: str, progress: ProgressCallback = lambda _: None) -> list[MediaItem]:
        if is_url(source):
            cached = self._load_cached_youtube_item(source)
            if cached is not None:
                if self.capture_video and cached.video_path is None:
                    try:
                        cached.video_path = self._download_visual(
                            source,
                            self.work_dir / cached.item_id,
                            cached.title,
                            progress,
                        )
                        self._save_youtube_metadata(self.work_dir / cached.item_id, cached)
                    except Exception as exc:  # noqa: BLE001 - audio remains usable
                        self.warnings.append(
                            f"{cached.title}：低解析度影像下載失敗，本次略過定時取幀：{exc}"
                        )
                progress(f"{cached.title}：使用媒體資訊快取")
                return [cached]
            return self._prepare_youtube(source, progress)
        return [self._prepare_local(Path(source), progress)]

    def _prepare_local(self, path: Path, progress: ProgressCallback) -> MediaItem:
        path = path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise MediaPreparationError(f"找不到檔案：{path}")
        if path.suffix.lower() not in self.SUPPORTED_LOCAL:
            raise MediaPreparationError(f"不支援的檔案格式：{path.suffix}")

        subtitle_path = (
            None if self.transcription_source == "whisper" else self._find_sidecar_subtitle(path)
        )

        item_id = stable_id(str(path))
        item_dir = self.work_dir / item_id
        legacy_audio = item_dir / "audio.wav"
        audio_path: Path | None = legacy_audio if legacy_audio.exists() else path
        if subtitle_path is not None:
            audio_path = None
        else:
            progress(f"{path.name}：直接讀取原始影音，不另行轉檔")

        return MediaItem(
            source=str(path),
            item_id=item_id,
            title=path.stem,
            source_type="file",
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            subtitle_kind="sidecar" if subtitle_path else None,
            video_path=(
                path if path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm"} else None
            ),
        )

    def _prepare_youtube(self, url: str, progress: ProgressCallback) -> list[MediaItem]:
        try:
            import yt_dlp
        except ImportError as exc:
            raise MediaPreparationError("尚未安裝 yt-dlp，無法處理 YouTube。") from exc
        progress("讀取 YouTube 資訊")
        with yt_dlp.YoutubeDL(
            {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
        ) as ydl:
            info = ydl.extract_info(url, download=False)
        raw_entries = info.get("entries")
        is_playlist = raw_entries is not None
        entries = [entry for entry in (raw_entries or [info]) if entry]
        items: list[MediaItem] = []
        for index, entry in enumerate(entries, start=1):
            video_url = entry.get("webpage_url") or entry.get("url")
            if not video_url or not str(video_url).startswith("http"):
                video_id = entry.get("id") or video_url
                video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else url
            entry_title = str(entry.get("title") or entry.get("id") or video_url)
            progress(f"播放清單 {index}/{len(entries)}：{entry_title}")
            try:
                if is_playlist:
                    item = self._download_youtube_item(str(video_url), progress)
                else:
                    item = self._download_youtube_item(str(video_url), progress, initial_info=info)
                items.append(item)
            except MediaPreparationError as exc:
                if not is_playlist:
                    raise
                warning = f"播放清單第 {index} 項「{entry_title}」下載失敗，已跳過：{exc}"
                self.warnings.append(warning)
                progress(warning)

        if not items:
            raise MediaPreparationError("播放清單中沒有任何影片可成功下載。")
        return items

    def _download_youtube_item(
        self,
        url: str,
        progress: ProgressCallback,
        initial_info: dict | None = None,
    ) -> MediaItem:
        import yt_dlp

        last_error: Exception | None = None
        for attempt in range(1, self.YOUTUBE_DOWNLOAD_ATTEMPTS + 1):
            try:
                return self._download_youtube_item_once(
                    url,
                    progress,
                    info=initial_info if attempt == 1 else None,
                )
            except (yt_dlp.utils.DownloadError, OSError, MediaPreparationError) as exc:
                last_error = exc
                if attempt < self.YOUTUBE_DOWNLOAD_ATTEMPTS:
                    progress(
                        "YouTube 下載失敗，正在重新取得串流網址後重試 "
                        f"({attempt + 1}/{self.YOUTUBE_DOWNLOAD_ATTEMPTS})"
                    )

        raise MediaPreparationError(
            f"已重試 {self.YOUTUBE_DOWNLOAD_ATTEMPTS} 次仍無法下載：{last_error}"
        ) from last_error

    def _download_youtube_item_once(
        self,
        url: str,
        progress: ProgressCallback,
        info: dict | None = None,
    ) -> MediaItem:
        import yt_dlp

        if info is None:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(url, download=False)
        video_id = str(info.get("id") or stable_id(url))
        item_id = f"youtube-{re.sub(r'[^A-Za-z0-9_-]', '_', video_id)}"
        title = str(info.get("title") or video_id)
        item_dir = self.work_dir / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        subtitle_path: Path | None = None
        subtitle_kind: str | None = None
        if self.transcription_source != "whisper":
            subtitle_path, subtitle_kind = self._download_preferred_subtitle(
                info, url, item_dir, progress
            )

        audio_path = self._find_downloaded_audio(item_dir)
        if subtitle_path is None and audio_path is None:
            progress(f"{title}：下載音訊")
            options = {
                "quiet": True,
                "no_warnings": True,
                "format": "bestaudio[abr<=128]/bestaudio/best",
                "outtmpl": str(item_dir / "source.%(ext)s"),
                "noplaylist": True,
                "retries": 10,
                "fragment_retries": 10,
                "extractor_retries": 3,
                "socket_timeout": 30,
            }
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
            audio_path = self._find_downloaded_audio(item_dir)
            if audio_path is None:
                raise MediaPreparationError(f"YouTube 音訊下載後找不到輸出：{title}")

        video_path = self._find_downloaded_video(item_dir)
        if self.capture_video and video_path is None:
            video_path = self._download_visual(url, item_dir, title, progress)

        item = MediaItem(
            source=url,
            item_id=item_id,
            title=title,
            source_type="youtube",
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            subtitle_kind=subtitle_kind,
            webpage_url=str(info.get("webpage_url") or url),
            video_path=video_path,
        )
        try:
            self._save_youtube_metadata(item_dir, item)
        except OSError as exc:
            self.warnings.append(f"{title}：媒體資訊快取寫入失敗：{exc}")
        return item

    def _load_cached_youtube_item(self, url: str) -> MediaItem | None:
        video_id = self._youtube_video_id(url)
        if not video_id:
            return None
        item_id = f"youtube-{re.sub(r'[^A-Za-z0-9_-]', '_', video_id)}"
        item_dir = self.work_dir / item_id
        metadata_path = item_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            import json

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            title = str(payload["title"])
            subtitle_path = self._cached_child_path(item_dir, payload.get("subtitle_filename"))
            audio_path = self._cached_child_path(item_dir, payload.get("audio_filename"))
            video_path = self._cached_child_path(item_dir, payload.get("video_filename"))
            if self.transcription_source == "whisper":
                subtitle_path = None
                if audio_path is None:
                    return None
            elif subtitle_path is not None:
                audio_path = None
            elif audio_path is None:
                return None
            return MediaItem(
                source=url,
                item_id=item_id,
                title=title,
                source_type="youtube",
                audio_path=audio_path,
                subtitle_path=subtitle_path,
                subtitle_kind=(str(payload.get("subtitle_kind")) if subtitle_path else None),
                webpage_url=str(payload.get("webpage_url") or url),
                video_path=video_path,
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _download_visual(
        self,
        url: str,
        item_dir: Path,
        title: str,
        progress: ProgressCallback,
    ) -> Path | None:
        import yt_dlp

        progress(f"{title}：下載低解析度影像供定時取幀")
        options = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestvideo[height<=480]/best[height<=480]/worstvideo",
            "outtmpl": str(item_dir / "visual.%(ext)s"),
            "noplaylist": True,
            "retries": 5,
            "socket_timeout": 30,
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
        video_path = self._find_downloaded_video(item_dir)
        if video_path is None:
            raise MediaPreparationError(f"YouTube 影像下載後找不到輸出：{title}")
        return video_path

    @staticmethod
    def _save_youtube_metadata(item_dir: Path, item: MediaItem) -> None:
        atomic_write_json(
            item_dir / "metadata.json",
            {
                "title": item.title,
                "webpage_url": item.webpage_url,
                "audio_filename": item.audio_path.name if item.audio_path else None,
                "video_filename": item.video_path.name if item.video_path else None,
                "subtitle_filename": (item.subtitle_path.name if item.subtitle_path else None),
                "subtitle_kind": item.subtitle_kind,
            },
        )

    @staticmethod
    def _cached_child_path(item_dir: Path, filename: object) -> Path | None:
        if not isinstance(filename, str) or not filename:
            return None
        path = item_dir / Path(filename).name
        return path if path.is_file() else None

    @staticmethod
    def _youtube_video_id(url: str) -> str | None:
        parsed = urlparse(url)
        host = parsed.netloc.lower().split(":", 1)[0]
        if host in {"youtu.be", "www.youtu.be"}:
            return parsed.path.strip("/").split("/", 1)[0] or None
        if host.endswith("youtube.com"):
            values = parse_qs(parsed.query).get("v") or []
            if values:
                return values[0]
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0].lower() in {"shorts", "live", "embed"}:
                return parts[1]
        return None

    def _download_preferred_subtitle(
        self, info: dict, url: str, item_dir: Path, progress: ProgressCallback
    ) -> tuple[Path | None, str | None]:
        manual = info.get("subtitles") or {}
        automatic = info.get("automatic_captions") or {}
        if manual:
            path = self._download_subtitle(info, url, item_dir, progress, manual, "manual")
            if path:
                return path, "manual"
        if automatic:
            path = self._download_subtitle(info, url, item_dir, progress, automatic, "automatic")
            if path:
                return path, "automatic"
        return None, None

    def _download_subtitle(
        self,
        info: dict,
        url: str,
        item_dir: Path,
        progress: ProgressCallback,
        subtitles: dict,
        kind: str,
    ) -> Path | None:
        language = self._choose_subtitle_language(list(subtitles))
        existing = list(item_dir.glob(f"{kind}.{language}.*"))
        if existing:
            return existing[0]
        label = "人工字幕" if kind == "manual" else "YouTube 自動字幕"
        progress(f"{info.get('title', 'YouTube')}：下載{label} ({language})")
        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": kind == "manual",
            "writeautomaticsub": kind == "automatic",
            "subtitleslangs": [language],
            "subtitlesformat": "vtt/best",
            "outtmpl": str(item_dir / f"{kind}.%(ext)s"),
            "noplaylist": True,
        }
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
        except (yt_dlp.utils.DownloadError, OSError):
            return None
        candidates = sorted(
            item_dir.glob(f"{kind}.*"),
            key=lambda path: path.stat().st_size,
            reverse=True,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _find_downloaded_audio(item_dir: Path) -> Path | None:
        legacy = item_dir / "audio.wav"
        if legacy.exists():
            return legacy
        candidates = [
            path
            for path in item_dir.glob("source.*")
            if path.is_file() and path.suffix.lower() not in {".part", ".ytdl"}
        ]
        return max(candidates, key=lambda path: path.stat().st_size) if candidates else None

    @staticmethod
    def _find_downloaded_video(item_dir: Path) -> Path | None:
        candidates = [
            path
            for path in item_dir.glob("visual.*")
            if path.is_file() and path.suffix.lower() not in {".part", ".ytdl"}
        ]
        return max(candidates, key=lambda path: path.stat().st_size) if candidates else None

    @staticmethod
    def _choose_subtitle_language(languages: list[str]) -> str:
        priorities = ("zh-Hant", "zh-TW", "zh", "en", "ja", "ko")
        lowered = {language.lower(): language for language in languages}
        for wanted in priorities:
            if wanted.lower() in lowered:
                return lowered[wanted.lower()]
        for wanted in priorities:
            for language in languages:
                if language.lower().startswith(f"{wanted.lower()}-"):
                    return language
        return languages[0]

    @staticmethod
    def _find_sidecar_subtitle(path: Path) -> Path | None:
        for suffix in (".vtt", ".srt"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                return candidate
        return None
