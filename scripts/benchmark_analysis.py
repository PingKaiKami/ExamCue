from __future__ import annotations

import argparse
import json
from pathlib import Path

from examcue.analysis import chunk_segments, focused_chunk_segments
from examcue.domain import TranscriptSegment

PROFILES = {
    "fine": (60, 2000),
    "standard": (120, 4000),
    "long": (240, 8000),
}


def load_segments(path: Path) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "start" in record:
            segments.append(TranscriptSegment(**record))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate ExamCue LLM calls without contacting an LLM server."
    )
    parser.add_argument("project", type=Path, help="ExamCue project directory")
    args = parser.parse_args()
    transcript_dir = args.project / "transcripts"
    paths = sorted(transcript_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No JSONL transcripts found in {transcript_dir}")

    print(
        "profile\tfull_calls\tfocused_calls\tcall_reduction\t"
        "full_chars\tfocused_chars\tchar_reduction"
    )
    for profile, (seconds, chars) in PROFILES.items():
        full_calls = 0
        focused_calls = 0
        full_chars = 0
        focused_chars = 0
        for path in paths:
            segments = load_segments(path)
            chunks = list(chunk_segments(segments, seconds, chars))
            focused_chunks = list(focused_chunk_segments(segments, seconds, chars))
            full_calls += len(chunks)
            focused_calls += len(focused_chunks)
            full_chars += sum(len(segment.text) for chunk in chunks for segment in chunk)
            focused_chars += sum(
                len(segment.text) for chunk in focused_chunks for segment in chunk
            )
        call_reduction = 0 if not full_calls else (1 - focused_calls / full_calls) * 100
        char_reduction = 0 if not full_chars else (1 - focused_chars / full_chars) * 100
        print(
            f"{profile}\t{full_calls}\t{focused_calls}\t{call_reduction:.1f}%\t"
            f"{full_chars}\t{focused_chars}\t{char_reduction:.1f}%"
        )


if __name__ == "__main__":
    main()
