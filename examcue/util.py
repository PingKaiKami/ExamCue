from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def find_executable(name: str, project_root: Path | None = None) -> str | None:
    if project_root:
        filename = name if name.lower().endswith(".exe") else f"{name}.exe"
        candidate = project_root / "tools" / filename
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def seconds_to_timestamp(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value.strip(), flags=re.IGNORECASE))
