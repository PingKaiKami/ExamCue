from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from examcue.pipeline import ExamCuePipeline, PipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run a completed ExamCue project with reference documents."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--server-executable", type=Path)
    parser.add_argument("--server-model", type=Path)
    parser.add_argument("--server-vendor-dir", type=Path)
    args = parser.parse_args()

    server: subprocess.Popen | None = None
    server_log = None
    if args.server_executable:
        if not args.server_model:
            parser.error("--server-model is required with --server-executable")
        server_env = os.environ.copy()
        if args.server_vendor_dir:
            server_env["PATH"] = (
                str(args.server_vendor_dir)
                + os.pathsep
                + server_env.get("PATH", "")
            )
        server_log = (args.target / "llama-server-analysis.log").open(
            "w", encoding="utf-8"
        )
        server = subprocess.Popen(
            [
                str(args.server_executable),
                "--model",
                str(args.server_model),
                "--alias",
                args.model,
                "--host",
                "127.0.0.1",
                "--port",
                "1235",
                "--ctx-size",
                "8192",
                "--parallel",
                "1",
                "--gpu-layers",
                "auto",
                "--fit",
                "on",
                "--reasoning",
                "off",
                "--no-reasoning-preserve",
                "--no-ui",
            ],
            cwd=args.server_executable.parent,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _attempt in range(150):
            if server.poll() is not None:
                raise RuntimeError("The local comparison LLM exited while loading")
            try:
                with urllib.request.urlopen(
                    f"{args.endpoint}/models", timeout=1
                ) as response:
                    if response.status == 200:
                        print("Local comparison LLM is ready", flush=True)
                        break
            except (OSError, TimeoutError, urllib.error.URLError):
                time.sleep(1)
        else:
            raise TimeoutError("The local comparison LLM did not become ready")

    baseline_metrics = json.loads(
        (args.baseline / "run_metrics.json").read_text(encoding="utf-8")
    )
    settings = baseline_metrics["settings"]
    sources = [record["source"] for record in baseline_metrics["sources"]]
    started = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[{time.perf_counter() - started:9.1f}s] {message}", flush=True)

    config = PipelineConfig(
        project_name=args.target.name,
        project_dir=args.target,
        sources=sources,
        output_language="繁體中文",
        whisper_model=settings["whisper_model"],
        transcription_source=settings["transcription_source"],
        transcription_mode=settings["transcription_mode"],
        analysis_chunk_seconds=int(settings["analysis_chunk_seconds"]),
        analysis_chunk_chars=int(settings["analysis_chunk_chars"]),
        analysis_scan_mode=settings["analysis_scan_mode"],
        llm_endpoint=args.endpoint,
        llm_model=args.model or settings["requested_llm_model"],
        use_llm=bool(settings["use_llm"]),
        reference_sources=args.reference,
        capture_frames=False,
        analyze_frames_with_llm=False,
        generate_quiz=False,
    )
    try:
        report = ExamCuePipeline(config, progress=progress).run()
        print(
            f"COMPLETE cues={len(report.cues)} warnings={len(report.warnings)} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        if server_log is not None:
            server_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
