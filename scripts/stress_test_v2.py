from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from examcue.media import expand_youtube_urls
from examcue.pipeline import ExamCuePipeline, PipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible ExamCue V2 stress test.")
    parser.add_argument("playlist")
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--frame-minutes", type=int, default=5)
    args = parser.parse_args()

    started = time.perf_counter()
    urls = expand_youtube_urls(args.playlist, lambda value: print(value, flush=True))
    if len(urls) != 15:
        print(f"Expected 15 playlist items, received {len(urls)}", file=sys.stderr)
        return 2

    args.project_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.project_dir / "stress_progress.log"

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - started
        line = f"[{elapsed:9.1f}s] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    config = PipelineConfig(
        project_name=args.project_dir.name,
        project_dir=args.project_dir,
        sources=urls,
        output_language="繁體中文",
        whisper_model="small",
        transcription_source="auto",
        transcription_mode="fast",
        analysis_chunk_seconds=120,
        analysis_chunk_chars=4000,
        analysis_scan_mode="focused",
        llm_endpoint=args.endpoint,
        llm_model=args.model,
        use_llm=True,
        reference_sources=args.reference,
        capture_frames=True,
        frame_interval_seconds=max(args.frame_minutes, 1) * 60,
        analyze_frames_with_llm=True,
        generate_quiz=True,
        max_quiz_questions=15,
    )
    report = ExamCuePipeline(config, progress=progress).run()
    print(
        f"Stress test complete: {len(report.cues)} cues, "
        f"{len(report.visual_observations)} frames, "
        f"{len(report.quiz_questions)} questions, "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
