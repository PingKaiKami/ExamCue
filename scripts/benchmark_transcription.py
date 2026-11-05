from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Faster-Whisper on an in-memory prefix of a media file."
    )
    parser.add_argument("media", type=Path, help="Audio or video file")
    parser.add_argument("--model", default="small", help="Whisper model name")
    parser.add_argument("--seconds", type=int, default=180, help="Prefix duration")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if not args.media.is_file():
        raise SystemExit(f"Media file not found: {args.media}")

    from examcue.transcription import configure_cuda_dll_directories

    configure_cuda_dll_directories()

    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio

    sample_rate = 16000
    decode_started = time.perf_counter()
    audio = decode_audio(str(args.media), sampling_rate=sample_rate)
    audio = audio[: max(args.seconds, 1) * sample_rate]
    decode_seconds = time.perf_counter() - decode_started
    duration = len(audio) / sample_rate

    load_started = time.perf_counter()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=8,
    )
    pipeline = BatchedInferencePipeline(model=model)
    load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    generated, info = pipeline.transcribe(
        audio,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments = list(generated)
    inference_seconds = time.perf_counter() - inference_started

    print(f"model={args.model}")
    print(f"compute_type={args.compute_type}")
    print(f"device={args.device}")
    print(f"batch_size={args.batch_size}")
    print(f"beam_size={args.beam_size}")
    print(f"audio_seconds={duration:.2f}")
    print(f"decode_seconds={decode_seconds:.2f}")
    print(f"model_load_seconds={load_seconds:.2f}")
    print(f"inference_seconds={inference_seconds:.2f}")
    print(f"realtime_factor={inference_seconds / max(duration, 0.001):.4f}")
    print(f"segments={len(segments)}")
    print(f"language={info.language}")


if __name__ == "__main__":
    main()
