from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .analysis import OpenAICompatibleClient
from .domain import MediaItem, VisualObservation
from .util import atomic_write_json

VISION_PROMPT_VERSION = "2"
_EMPTY_CACHE_VALUE = "__EXAMCUE_NO_VISIBLE_CONTENT__"


def capture_periodic_frames(
    item: MediaItem,
    output_dir: Path,
    interval_seconds: int = 300,
    duration_seconds: float = 0,
) -> list[VisualObservation]:
    """Capture course-video frames with PyAV, avoiding an OpenCV dependency."""
    video_path = item.video_path
    if video_path is None or not video_path.exists() or interval_seconds <= 0:
        return []
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Frame capture requires PyAV") from exc

    target = output_dir / item.item_id
    target.mkdir(parents=True, exist_ok=True)
    observations: list[VisualObservation] = []
    with av.open(str(video_path)) as container:
        stream = next((value for value in container.streams if value.type == "video"), None)
        if stream is None:
            return []
        if duration_seconds <= 0 and stream.duration is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        timestamps = [0.0]
        current = float(interval_seconds)
        while current < duration_seconds:
            timestamps.append(current)
            current += interval_seconds
        for index, timestamp in enumerate(timestamps, start=1):
            container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
            frame = next(container.decode(stream), None)
            if frame is None:
                continue
            image_path = target / f"frame-{index:04d}-{int(timestamp):06d}s.jpg"
            frame.to_image().convert("RGB").save(image_path, quality=86)
            observations.append(
                VisualObservation(
                    source_id=item.item_id,
                    source_title=item.title,
                    timestamp=timestamp,
                    image_path=image_path,
                )
            )
    return observations


def describe_frames(
    observations: list[VisualObservation],
    client: OpenAICompatibleClient,
    output_language: str,
    progress=lambda _: None,
    cache_dir: Path | None = None,
) -> list[str]:
    warnings: list[str] = []
    cache_path = cache_dir / "vision-descriptions.json" if cache_dir else None
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                cache = {
                    str(key): str(description)
                    for key, description in value.items()
                    if isinstance(description, str)
                }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            cache = {}
    representatives: list[int] = []
    hashes: list[int] = []
    duplicate_of: dict[int, int] = {}
    for index, observation in enumerate(observations):
        image_hash = _difference_hash(observation.image_path)
        match = next(
            (
                representative
                for representative, known_hash in zip(representatives, hashes, strict=True)
                if (image_hash ^ known_hash).bit_count() <= 4
            ),
            None,
        )
        if match is None:
            representatives.append(index)
            hashes.append(image_hash)
        else:
            duplicate_of[index] = match

    status = getattr(client, "status", None)
    if status is None and hasattr(client, "connect"):
        status = client.connect()
    model_identity = (
        f"{status.endpoint}|{status.model}"
        if status
        else f"{getattr(client, 'requested_endpoint', '')}|{getattr(client, 'requested_model', '')}"
    )
    cache_keys = {
        index: _vision_cache_key(
            observations[index].image_path,
            output_language,
            model_identity,
        )
        for index in representatives
    }
    uncached_representatives: list[int] = []
    processed_representatives: set[int] = set()
    cache_hits = 0
    for index in representatives:
        cache_key = cache_keys[index]
        if cache_key in cache:
            cached_description = cache[cache_key].strip()
            observations[index].description = (
                "" if cached_description == _EMPTY_CACHE_VALUE else cached_description
            )
            cache_hits += 1
        else:
            uncached_representatives.append(index)

    batch_size = 4
    failed_batches = 0
    for offset in range(0, len(uncached_representatives), batch_size):
        batch = uncached_representatives[offset : offset + batch_size]
        progress(
            f"畫面理解 {min(offset + len(batch), len(uncached_representatives))}/"
            f"{len(uncached_representatives)}（快取 {cache_hits} 張；"
            f"略過 {len(duplicate_of)} 張相似畫面）"
        )
        try:
            records = client.analyze_images(
                [observations[index].image_path for index in batch],
                f"Output language: {output_language}. Describe only what is visibly supported.",
            )
            for record in records:
                local_index = int(record.get("i", -1))
                if 0 <= local_index < len(batch):
                    observations[batch[local_index]].description = str(
                        record.get("d") or ""
                    ).strip()
            processed_representatives.update(batch)
        except Exception as exc:  # noqa: BLE001 - vision is optional enrichment
            # Some compatible endpoints accept one image but not a multi-image request.
            # Degrade only this batch so one incompatible response does not discard every
            # captured frame from a long playlist.
            failed_batches += 1
            single_failures = 0
            for index in batch:
                try:
                    records = client.analyze_image(
                        observations[index].image_path,
                        f"Output language: {output_language}. Describe only what is visibly supported.",
                    )
                    if records:
                        observations[index].description = str(records[0].get("d") or "").strip()
                    processed_representatives.add(index)
                except Exception:  # noqa: BLE001 - optional per-frame enrichment
                    single_failures += 1
            if single_failures == len(batch):
                warnings.append(f"目前模型無法分析部分影片畫面，取幀檔案仍已保留：{exc}")
                # A model-level incompatibility will repeat for every batch. Stop after
                # the first completely failed batch, while retaining captured images.
                break
    for index, representative in duplicate_of.items():
        observations[index].description = observations[representative].description
    if cache_path:
        for index in processed_representatives:
            cache[cache_keys[index]] = observations[index].description or _EMPTY_CACHE_VALUE
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cache_path, cache)
        except OSError as exc:
            warnings.append(f"畫面描述快取寫入失敗：{exc}")
    if failed_batches and any(item.description for item in observations):
        warnings.append(f"有 {failed_batches} 個多圖批次改以單張模式完成，畫面分析結果仍可使用。")
    return warnings


def _vision_cache_key(
    image_path: Path,
    output_language: str,
    model_identity: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(VISION_PROMPT_VERSION.encode("utf-8"))
    digest.update(output_language.encode("utf-8"))
    digest.update(model_identity.encode("utf-8"))
    digest.update(image_path.read_bytes())
    return digest.hexdigest()


def _difference_hash(image_path: Path) -> int:
    from PIL import Image

    with Image.open(image_path) as image:
        resized = image.convert("L").resize((9, 8))
        pixels = [resized.getpixel((column, row)) for row in range(8) for column in range(9)]
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value
