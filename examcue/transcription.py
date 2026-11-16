from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path

from .domain import MediaItem, TranscriptSegment
from .util import normalize_text

ProgressCallback = Callable[[str], None]
ControlCallback = Callable[[], None]

TRANSCRIPTION_PROFILES = {
    "fast": {"beam_size": 1, "batch_size": 8},
    "standard": {"beam_size": 3, "batch_size": 4},
    "accurate": {"beam_size": 5, "batch_size": 1},
}

# Sets only the script and punctuation style; course terms belong in hotwords.
ZH_INITIAL_PROMPT = "以下是繁體中文的課堂講解。"
# Part of the transcript cache key so transcripts made before the prompt are redone.
WHISPER_CACHE_TAG = "zh-prompt1"

TRANSCRIPTION_METHOD_LABELS = {
    "manual_subtitle": "人工字幕",
    "automatic_subtitle": "YouTube 自動字幕",
    "sidecar_subtitle": "同名字幕檔",
    "whisper": "Whisper 語音辨識",
}

_CUDA_DLL_HANDLES: dict[str, object] = {}


def _nvidia_wheel_bin_directories() -> list[Path]:
    directories: list[Path] = []
    for module_name in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            candidate = Path(location) / "bin"
            if candidate.is_dir() and candidate not in directories:
                directories.append(candidate)
    return directories


def configure_cuda_dll_directories() -> list[Path]:
    """Make NVIDIA wheel DLLs discoverable without changing the system PATH."""
    if sys.platform != "win32":
        return []
    directories = _nvidia_wheel_bin_directories()
    for path in directories:
        value = str(path)
        key = value.casefold()
        if hasattr(os, "add_dll_directory") and key not in _CUDA_DLL_HANDLES:
            try:
                _CUDA_DLL_HANDLES[key] = os.add_dll_directory(value)
            except OSError:
                pass
    current_path = os.environ.get("PATH", "")
    existing = {part.lower() for part in current_path.split(os.pathsep) if part}
    missing = [str(path) for path in directories if str(path).lower() not in existing]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, current_path])
    return directories


def cuda_diagnostics() -> dict:
    directories = configure_cuda_dll_directories()
    result = {
        "platform": sys.platform,
        "gpu_count": 0,
        "gpu_name": "",
        "cuda_ready": False,
        "device": "cpu",
        "compute_type": "int8",
        "dll_directories": [str(path) for path in directories],
        "missing_dlls": [],
        "error": "",
    }
    try:
        import ctranslate2

        result["ctranslate2_version"] = ctranslate2.__version__
        result["gpu_count"] = ctranslate2.get_cuda_device_count()
    except (ImportError, OSError, RuntimeError) as exc:
        result["error"] = str(exc)
        return result

    if sys.platform == "win32":
        for filename in ("cublas64_12.dll", "cudnn64_9.dll"):
            try:
                ctypes.WinDLL(filename)
            except OSError:
                result["missing_dlls"].append(filename)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        if completed.returncode == 0:
            result["gpu_name"] = completed.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    result["cuda_ready"] = bool(
        result["gpu_count"] and not result["missing_dlls"] and not result["error"]
    )
    if result["cuda_ready"]:
        result["device"] = "cuda"
        result["compute_type"] = "float16"
    return result


def test_cuda_transcription(model_name: str = "small") -> dict:
    """Load the selected model and execute one short GPU inference."""
    diagnostics = cuda_diagnostics()
    if not diagnostics["cuda_ready"]:
        missing = ", ".join(diagnostics["missing_dlls"]) or diagnostics["error"]
        raise TranscriptionError(f"CUDA 執行環境尚未就緒：{missing or '未偵測到 GPU'}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("尚未安裝 faster-whisper") from exc

    started = time.perf_counter()
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name
        with wave.open(temp_path, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            frames = bytearray()
            for index in range(16000):
                sample = int(1000 * math.sin(2 * math.pi * 440 * index / 16000))
                frames.extend(struct.pack("<h", sample))
            audio.writeframes(frames)
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        segments, _info = model.transcribe(
            temp_path,
            language="zh",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        list(segments)
    except (OSError, RuntimeError) as exc:
        raise TranscriptionError(f"GPU 辨識測試失敗：{exc}") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
    diagnostics["test_seconds"] = round(time.perf_counter() - started, 3)
    diagnostics["tested_model"] = model_name
    return diagnostics


class TranscriptionError(RuntimeError):
    pass


class TranscriptStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, item: MediaItem) -> Path:
        return self.directory / f"{item.item_id}.jsonl"

    def load(
        self, item: MediaItem, expected_model: str | None = None
    ) -> list[TranscriptSegment] | None:
        path = self.path_for(item)
        if not path.exists():
            return None
        segments: list[TranscriptSegment] = []
        metadata: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if "meta" in data:
                metadata = data["meta"]
            elif "start" in data:
                segments.append(TranscriptSegment(**data))
        if expected_model is not None and metadata.get("model") != expected_model:
            return None
        return segments

    def save(
        self,
        item: MediaItem,
        segments: list[TranscriptSegment],
        language: str,
        model_name: str,
    ) -> Path:
        path = self.path_for(item)
        temp = path.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            metadata = {"meta": {"language": language, "model": model_name}}
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for segment in segments:
                handle.write(json.dumps(segment.to_dict(), ensure_ascii=False) + "\n")
        temp.replace(path)
        return path


class Transcriber:
    def __init__(
        self,
        store: TranscriptStore,
        model_name: str = "small",
        mode: str = "standard",
        hotwords: str = "",
    ):
        self.store = store
        self.model_name = model_name
        self.mode = mode if mode in TRANSCRIPTION_PROFILES else "standard"
        self.hotwords = hotwords.strip()
        self._model = None
        self._batched_model = None
        self._device = ""
        self._compute_type = ""
        self.last_method = "whisper"

    @property
    def profile(self) -> dict[str, int]:
        return TRANSCRIPTION_PROFILES[self.mode]

    @property
    def device(self) -> str:
        return self._device or "cache"

    @property
    def compute_type(self) -> str:
        return self._compute_type or "cache"

    def transcribe(
        self,
        item: MediaItem,
        progress: ProgressCallback = lambda _: None,
        control: ControlCallback = lambda: None,
    ) -> tuple[list[TranscriptSegment], str]:
        control()
        if item.subtitle_path:
            subtitle_kind = item.subtitle_kind or "sidecar"
            expected_model = f"subtitle:{subtitle_kind}"
            self.last_method = f"{subtitle_kind}_subtitle"
        else:
            if self.hotwords:
                hotword_key = hashlib.sha256(self.hotwords.encode("utf-8")).hexdigest()[:16]
                expected_model = (
                    f"{self.model_name}:{self.mode}:{WHISPER_CACHE_TAG}:hotwords={hotword_key}"
                )
            else:
                expected_model = f"{self.model_name}:{self.mode}:{WHISPER_CACHE_TAG}"
            self.last_method = "whisper"
        cached = self.store.load(item, expected_model=expected_model)
        if cached is not None:
            progress(f"{item.title}：使用逐字稿快取")
            return cached, "cached"

        if item.subtitle_path:
            method_label = TRANSCRIPTION_METHOD_LABELS.get(self.last_method, "現有字幕")
            progress(f"{item.title}：使用{method_label}")
            segments = parse_subtitle(item.subtitle_path)
            if segments:
                self.store.save(item, segments, "subtitle", expected_model)
                return segments, "subtitle"

        if not item.audio_path:
            raise TranscriptionError(f"{item.title} 沒有可用音訊")
        progress(
            f"{item.title}：本機語音辨識 "
            f"({self.model_name} / {self.mode} / batch {self.profile['batch_size']})"
        )
        try:
            segments, language = self._transcribe_audio(
                item.audio_path, progress=progress, control=control
            )
        except (OSError, RuntimeError) as exc:
            if self._device != "cuda" or not _is_cuda_runtime_error(exc):
                raise
            progress(f"{item.title}：CUDA 執行環境不完整，改用 CPU 重新辨識")
            self._model = None
            self._batched_model = None
            self._device = "cpu"
            self._compute_type = "int8"
            segments, language = self._transcribe_audio(
                item.audio_path, force_cpu=True, progress=progress, control=control
            )
        control()
        self.store.save(item, segments, language, expected_model)
        return segments, language

    def _transcribe_audio(
        self,
        audio_path: Path,
        force_cpu: bool = False,
        progress: ProgressCallback = lambda _: None,
        control: ControlCallback = lambda: None,
    ) -> tuple[list[TranscriptSegment], str]:
        control()
        model = self._get_model(force_cpu=force_cpu)
        batch_size = self.profile["batch_size"]
        options = {
            "language": None,
            "beam_size": self.profile["beam_size"],
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
        }
        if self.hotwords:
            options["hotwords"] = self.hotwords
        use_batched = batch_size > 1

        def run():
            nonlocal use_batched
            if use_batched:
                batched_model = self._get_batched_model(model)
                try:
                    return batched_model.transcribe(
                        str(audio_path), batch_size=batch_size, **options
                    )
                except TypeError as exc:
                    if "hotwords" not in options:
                        raise
                    progress(f"目前語音引擎不支援教材術語提示，略過提示繼續：{exc}")
                    options.pop("hotwords", None)
                    return batched_model.transcribe(
                        str(audio_path), batch_size=batch_size, **options
                    )
                except RuntimeError:
                    progress("批次辨識無法執行，改用單批模式繼續")
                    use_batched = False
                    return model.transcribe(
                        str(audio_path), condition_on_previous_text=True, **options
                    )
            try:
                return model.transcribe(
                    str(audio_path), condition_on_previous_text=True, **options
                )
            except TypeError as exc:
                if "hotwords" not in options:
                    raise
                progress(f"目前語音引擎不支援教材術語提示，略過提示繼續：{exc}")
                options.pop("hotwords", None)
                return model.transcribe(
                    str(audio_path), condition_on_previous_text=True, **options
                )

        # faster-whisper detects the language before decoding lazily, so restarting
        # here only repeats audio loading. Without a prompt Whisper mixes Simplified and
        # Traditional script in Mandarin lectures and often omits punctuation.
        generated, info = run()
        if info.language == "zh":
            options["language"] = "zh"
            options["initial_prompt"] = ZH_INITIAL_PROMPT
            generated, info = run()
        segments: list[TranscriptSegment] = []
        duration = max(float(getattr(info, "duration", 0) or 0), 0)
        last_reported_percent = -1
        for segment in generated:
            control()
            text = normalize_text(segment.text)
            if text:
                segments.append(
                    TranscriptSegment(
                        start=round(segment.start, 2),
                        end=round(segment.end, 2),
                        text=text,
                    )
                )
            if duration:
                percent = min(100, int(segment.end / duration * 100))
                if percent != last_reported_percent:
                    progress(f"語音辨識進度 {min(segment.end, duration):.2f}/{duration:.2f}")
                    last_reported_percent = percent
        return segments, str(info.language or "auto")

    def release(self) -> None:
        """Free the Whisper model so later LLM and vision work has the GPU memory."""
        self._model = None
        self._batched_model = None
        gc.collect()

    def _get_batched_model(self, model):
        if self._batched_model is None:
            try:
                from faster_whisper import BatchedInferencePipeline
            except ImportError as exc:
                raise TranscriptionError(
                    "目前的 faster-whisper 不支援批次辨識，請更新套件。"
                ) from exc
            self._batched_model = BatchedInferencePipeline(model=model)
        return self._batched_model

    def _get_model(self, force_cpu: bool = False):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError("尚未安裝 faster-whisper。") from exc
            self._device = "cpu" if force_cpu or not _cuda_runtime_available() else "cuda"
            self._compute_type = "int8" if self._device == "cpu" else "float16"
            self._model = WhisperModel(
                self.model_name,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=8,
            )
        return self._model


def _cuda_runtime_available() -> bool:
    return bool(cuda_diagnostics()["cuda_ready"])


def _is_cuda_runtime_error(error: BaseException) -> bool:
    message = str(error).lower()
    markers = ("cuda", "cublas", "cudnn", "curand", "nvrtc")
    return any(marker in message for marker in markers)


def parse_subtitle(path: Path) -> list[TranscriptSegment]:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\r?\n+", content)
    segments: list[TranscriptSegment] = []
    timing = re.compile(
        r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s+-->\s+"
        r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})"
    )
    for block in blocks:
        match = timing.search(block)
        if not match:
            continue
        lines = block[match.end() :].splitlines()
        text = normalize_text(
            re.sub(r"<[^>]+>", "", " ".join(line.strip() for line in lines if line.strip()))
        )
        if text:
            segment = TranscriptSegment(
                start=_parse_timestamp(match.group("start")),
                end=_parse_timestamp(match.group("end")),
                text=text,
            )
            if segments and segments[-1].text == segment.text:
                segments[-1].end = segment.end
            else:
                segments.append(segment)
    return segments


def _parse_timestamp(value: str) -> float:
    normalized = value.replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
