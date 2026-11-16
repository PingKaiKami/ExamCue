from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .analysis import ANALYSIS_PROMPT_VERSION, CueAnalyzer, deduplicate_cues
from .domain import (
    AnalysisReport,
    CueKind,
    ExamCue,
    MediaItem,
    TranscriptSegment,
    VisualObservation,
)
from .media import MediaPreparer
from .quiz import generate_quiz, write_quiz
from .references import ReferenceCorpus, load_reference_corpus
from .reporting import write_report
from .transcription import TRANSCRIPTION_METHOD_LABELS, Transcriber, TranscriptStore
from .util import atomic_write_json
from .vision import capture_periodic_frames, describe_frames

ProgressCallback = Callable[[str], None]
ANALYSIS_CACHE_VERSION = ANALYSIS_PROMPT_VERSION


class PipelineCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class PipelineConfig:
    project_name: str
    project_dir: Path
    sources: list[str]
    output_language: str = "繁體中文"
    whisper_model: str = "large-v3"
    transcription_source: str = "auto"
    transcription_mode: str = "standard"
    analysis_chunk_seconds: int = 120
    analysis_chunk_chars: int = 4000
    analysis_scan_mode: str = "full"
    llm_endpoint: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    use_llm: bool = True
    reference_sources: list[str] | None = None
    capture_frames: bool = False
    frame_interval_seconds: int = 300
    analyze_frames_with_llm: bool = True
    generate_quiz: bool = True
    max_quiz_questions: int = 10


class ExamCuePipeline:
    def __init__(
        self,
        config: PipelineConfig,
        progress: ProgressCallback = lambda _: None,
        cancel_event: threading.Event | None = None,
        resume_event: threading.Event | None = None,
    ):
        self.config = config
        self.progress = progress
        self.cancel_event = cancel_event or threading.Event()
        self.resume_event = resume_event or threading.Event()
        self.resume_event.set()
        self.media_dir = config.project_dir / "media"
        self.transcript_dir = config.project_dir / "transcripts"
        self.cache_dir = config.project_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: dict = {}

    def run(self) -> AnalysisReport:
        started_at = datetime.now().astimezone()
        started = time.perf_counter()
        self._metrics = {
            "schema_version": 1,
            "project_name": self.config.project_name,
            "started_at": started_at.isoformat(timespec="seconds"),
            "status": "running",
            "settings": {
                "source_count": len(self.config.sources),
                "whisper_model": self.config.whisper_model,
                "transcription_source": self.config.transcription_source,
                "transcription_mode": self.config.transcription_mode,
                "analysis_scan_mode": self.config.analysis_scan_mode,
                "analysis_chunk_seconds": self.config.analysis_chunk_seconds,
                "analysis_chunk_chars": self.config.analysis_chunk_chars,
                "use_llm": self.config.use_llm,
                "requested_llm_endpoint": self.config.llm_endpoint,
                "requested_llm_model": self.config.llm_model,
                "reference_count": len(self.config.reference_sources or []),
                "capture_frames": self.config.capture_frames,
                "frame_interval_seconds": self.config.frame_interval_seconds,
                "generate_quiz": self.config.generate_quiz,
            },
            "phase_seconds": {},
            "sources": [],
        }
        try:
            report = self._run()
        except PipelineCancelled:
            self._metrics["status"] = "cancelled"
            raise
        except Exception as exc:
            self._metrics["status"] = "failed"
            self._metrics["error"] = str(exc)
            raise
        else:
            self._metrics["status"] = "completed"
            return report
        finally:
            self._metrics["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            self._metrics["total_seconds"] = round(time.perf_counter() - started, 3)
            self._write_run_metrics()

    def _run(self) -> AnalysisReport:
        self.config.project_dir.mkdir(parents=True, exist_ok=True)
        media_transcription_started = time.perf_counter()
        reference_corpus = ReferenceCorpus([], "", "", [])
        warnings: list[str] = []
        if self.config.reference_sources:
            self.progress("正在解析投影片與教材…")
            try:
                reference_corpus = load_reference_corpus(
                    self.config.reference_sources,
                    cache_path=self.cache_dir / "reference-index.json",
                )
            except Exception as exc:  # noqa: BLE001 - references are optional enrichment
                warnings.append(f"教材解析失敗，已繼續分析課程：{exc}")
        store = TranscriptStore(self.transcript_dir)
        transcriber = Transcriber(
            store,
            self.config.whisper_model,
            mode=self.config.transcription_mode,
            # Course material is retrieval context for cue analysis. Applying one
            # global glossary to every audio segment can bias Whisper and invalidate
            # otherwise reusable transcript caches.
            hotwords="",
        )
        analyzer = CueAnalyzer(
            output_language=self.config.output_language,
            endpoint=self.config.llm_endpoint,
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            use_llm=self.config.use_llm,
            chunk_max_seconds=self.config.analysis_chunk_seconds,
            chunk_max_chars=self.config.analysis_chunk_chars,
            chunk_cache_dir=self.cache_dir / "analysis-chunks",
            scan_mode=self.config.analysis_scan_mode,
            reference_context=reference_corpus.context,
            reference_chunks=reference_corpus.chunks,
            reference_prompt_chars=1200,
        )

        transcribed_items: list[tuple[MediaItem, list[TranscriptSegment], str]] = []
        visual_observations: list[VisualObservation] = []
        source_records: list[dict] = []
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="examcue-download")
        futures = [
            executor.submit(
                self._prepare_source,
                source,
                number,
                len(self.config.sources),
            )
            for number, source in enumerate(self.config.sources, start=1)
        ]
        completed_normally = False
        item_number = 0
        media_preparation_sum = 0.0
        try:
            for future in futures:
                self._check_control()
                prepared_items, source_warnings, preparation_seconds = future.result()
                media_preparation_sum += preparation_seconds
                warnings.extend(source_warnings)
                for item in prepared_items:
                    item_number += 1
                    total_hint = max(len(self.config.sources), item_number)
                    self._check_control()
                    self.progress(f"[語音辨識 {item_number}/{total_hint}] {item.title}")
                    try:
                        transcription_started = time.perf_counter()
                        segments, detected_language = transcriber.transcribe(
                            item, self.progress, self._check_control
                        )
                        transcription_seconds = time.perf_counter() - transcription_started
                        self._check_control()
                        transcribed_items.append((item, segments, detected_language))
                        if self.config.capture_frames:
                            try:
                                observations = capture_periodic_frames(
                                    item,
                                    self.config.project_dir / "frames",
                                    self.config.frame_interval_seconds,
                                    max((segment.end for segment in segments), default=0.0),
                                )
                                visual_observations.extend(observations)
                                self.progress(
                                    f"{item.title}：已擷取 {len(observations)} 張課程畫面"
                                )
                            except Exception as exc:  # noqa: BLE001
                                warnings.append(f"{item.title}：畫面擷取失敗：{exc}")
                        source_records.append(
                            {
                                "item_id": item.item_id,
                                "title": item.title,
                                "source": item.source,
                                "source_type": item.source_type,
                                "webpage_url": item.webpage_url,
                                "detected_language": detected_language,
                                "transcription_method": transcriber.last_method,
                                "transcription_method_label": (
                                    TRANSCRIPTION_METHOD_LABELS.get(
                                        transcriber.last_method,
                                        transcriber.last_method,
                                    )
                                ),
                                "transcript_segments": len(segments),
                                "duration_seconds": round(
                                    max((segment.end for segment in segments), default=0.0),
                                    3,
                                ),
                                "transcription_seconds": round(transcription_seconds, 3),
                                "used_transcript_cache": detected_language == "cached",
                                "transcription_device": transcriber.device,
                                "transcription_compute_type": transcriber.compute_type,
                                "transcription_beam_size": transcriber.profile["beam_size"],
                                "transcription_batch_size": transcriber.profile["batch_size"],
                            }
                        )
                        self.progress(f"語音辨識完成 {item_number}/{total_hint}")
                    except PipelineCancelled:
                        raise
                    except Exception as exc:  # noqa: BLE001 - isolate media failures
                        warnings.append(f"{item.title}：處理失敗：{exc}")
            completed_normally = True
        finally:
            executor.shutdown(
                wait=True,
                cancel_futures=not completed_normally,
            )
            transcriber.release()

        self._metrics["phase_seconds"]["media_and_transcription_wall"] = round(
            time.perf_counter() - media_transcription_started, 3
        )
        self._metrics["phase_seconds"]["media_preparation_sum"] = round(media_preparation_sum, 3)
        self._metrics["phase_seconds"]["transcription_sum"] = round(
            sum(float(source["transcription_seconds"]) for source in source_records),
            3,
        )

        self.progress("語音辨識階段完成")
        if visual_observations and self.config.use_llm and self.config.analyze_frames_with_llm:
            warnings.extend(
                describe_frames(
                    visual_observations,
                    analyzer.client,
                    self.config.output_language,
                    self.progress,
                    self.cache_dir,
                )
            )
            descriptions = [
                f"[{item.source_title} {item.timestamp:.0f}s] {item.description}"
                for item in visual_observations
                if item.description
            ]
            if descriptions:
                frame_context = (
                    "\n\nPeriodically captured lecture-frame descriptions "
                    "(context only; not exam evidence):\n" + "\n".join(descriptions)
                )
                reference_limit = 120000
                retained_reference = analyzer.reference_context[
                    : max(reference_limit - len(frame_context), 0)
                ]
                analyzer.reference_context = (retained_reference + frame_context)[:reference_limit]
        analysis_started = time.perf_counter()
        all_cues: list[ExamCue] = []
        analysis_work = [
            max(analyzer.estimate_work(segments), 1)
            for _item, segments, _detected_language in transcribed_items
        ]
        total_analysis_work = max(sum(analysis_work), 1)
        completed_analysis_work = 0
        self.progress(f"分析總工作量 {total_analysis_work}")
        for number, (item, segments, _detected_language) in enumerate(transcribed_items, start=1):
            self._check_control()
            self.progress(f"[考點分析 {number}/{len(transcribed_items)}] {item.title}")
            item_work = analysis_work[number - 1]
            item_analysis_started = time.perf_counter()
            used_analysis_cache = False
            try:
                cues = self._load_analysis_cache(item)
                if cues is None:

                    def analysis_progress(
                        message: str,
                        *,
                        item_work: int = item_work,
                        completed_work: int = completed_analysis_work,
                    ) -> None:
                        self.progress(message)
                        match = re.fullmatch(r"分析工作進度\s+(\d+)/(\d+)", message)
                        if match:
                            current = min(int(match.group(1)), item_work)
                            self.progress(
                                f"整體分析工作進度 {completed_work + current}/{total_analysis_work}"
                            )

                    cues = analyzer.analyze_item(
                        item, segments, analysis_progress, self._check_control
                    )
                    try:
                        self._save_analysis_cache(item, cues)
                    except OSError as exc:
                        warnings.append(f"{item.title}：分析快取寫入失敗：{exc}")
                else:
                    used_analysis_cache = True
                    self.progress(f"{item.title}：使用分析快取")
                all_cues.extend(cues)
            except PipelineCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - isolate failures by media item
                warnings.append(f"{item.title}：考點分析失敗：{exc}")
                cues = []
            source_record = next(
                (record for record in source_records if record["item_id"] == item.item_id),
                None,
            )
            if source_record is not None:
                source_record["analysis_seconds"] = round(
                    time.perf_counter() - item_analysis_started, 3
                )
                source_record["used_analysis_cache"] = used_analysis_cache
                source_record["cue_count_before_global_deduplication"] = len(cues)
            completed_analysis_work += item_work
            self.progress(f"整體分析工作進度 {completed_analysis_work}/{total_analysis_work}")
            self.progress(f"考點分析完成 {number}/{len(transcribed_items)}")

        self._metrics["phase_seconds"]["exam_cue_analysis"] = round(
            time.perf_counter() - analysis_started, 3
        )
        warnings.extend(analyzer.warnings)
        deduplicated_cues = deduplicate_cues(all_cues)
        report = AnalysisReport(
            project_name=self.config.project_name,
            output_language=self.config.output_language,
            cues=deduplicated_cues,
            sources=source_records,
            warnings=list(dict.fromkeys(warnings)),
            visual_observations=visual_observations,
            reference_documents=reference_corpus.documents,
        )
        quiz_started = time.perf_counter()
        if self.config.generate_quiz and self.config.use_llm and deduplicated_cues:
            try:
                self.progress("正在依考點生成選擇題…")
                report.quiz_questions = generate_quiz(
                    deduplicated_cues,
                    analyzer.client,
                    self.config.output_language,
                    self.config.max_quiz_questions,
                    self.cache_dir,
                )
                write_quiz(report.quiz_questions, self.config.project_dir)
            except Exception as exc:  # noqa: BLE001 - report remains useful without quiz
                report.warnings.append(f"選擇題生成失敗：{exc}")
        self._metrics["phase_seconds"]["quiz_generation"] = round(
            time.perf_counter() - quiz_started, 3
        )
        reporting_started = time.perf_counter()
        write_report(report, self.config.project_dir)
        self._metrics["phase_seconds"]["report_generation"] = round(
            time.perf_counter() - reporting_started, 3
        )
        self._metrics["total_media_duration_seconds"] = round(
            sum(float(source["duration_seconds"]) for source in source_records), 3
        )
        self._metrics["media_item_count"] = len(source_records)
        self._metrics["cue_count"] = len(deduplicated_cues)
        self._metrics["reference_document_count"] = len(reference_corpus.documents)
        self._metrics["reference_index_chunk_count"] = len(reference_corpus.chunks)
        self._metrics["reference_prompt_character_limit"] = 1200
        self._metrics["captured_frame_count"] = len(visual_observations)
        self._metrics["described_frame_count"] = sum(
            bool(item.description) for item in visual_observations
        )
        self._metrics["quiz_question_count"] = len(report.quiz_questions)
        self._metrics["sources"] = source_records
        self._metrics["llm"] = {
            "enabled": self.config.use_llm,
            "endpoint": analyzer.client.status.endpoint if analyzer.client.status else "",
            "model": analyzer.client.status.model if analyzer.client.status else "",
            "analysis_request_count": analyzer.client.analysis_request_count,
            "analysis_request_seconds": round(analyzer.client.analysis_request_seconds, 3),
            "chunk_count": analyzer.analysis_chunk_count,
            "chunk_cache_hits": analyzer.analysis_chunk_cache_hits,
            "failure_count": analyzer.llm_failure_count,
            "non_llm_chunk_count": analyzer.fallback_chunk_count,
            "quality_filtered_segment_count": analyzer.quality_filtered_segment_count,
            "quality_rejected_cue_count": analyzer.quality_rejected_cue_count,
            "empty_direct_response_count": analyzer.empty_direct_response_count,
        }
        self.progress(f"完成：{self.config.project_dir / 'report.md'}")
        return report

    def _prepare_source(
        self, source: str, number: int, total: int
    ) -> tuple[list[MediaItem], list[str], float]:
        started = time.perf_counter()
        preparer = MediaPreparer(
            self.media_dir,
            Path.cwd(),
            transcription_source=self.config.transcription_source,
            capture_video=self.config.capture_frames,
        )
        warnings: list[str] = []
        try:
            self.progress(f"[媒體準備 {number}/{total}] {source}")
            items = preparer.prepare(source, self.progress)
        except Exception as exc:  # noqa: BLE001 - isolate failures by source
            items = []
            warnings.append(f"{source}：媒體準備失敗：{exc}")
        finally:
            warnings.extend(preparer.pop_warnings())
        return items, warnings, time.perf_counter() - started

    def _write_run_metrics(self) -> None:
        try:
            atomic_write_json(self.config.project_dir / "run_metrics.json", self._metrics)
            history_path = self.config.project_dir / "run_history.jsonl"
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self._metrics, ensure_ascii=False) + "\n")
        except OSError:
            # Performance logging must never turn a successful analysis into a failure.
            pass

    def _check_control(self) -> None:
        while not self.resume_event.wait(0.1):
            if self.cancel_event.is_set():
                raise PipelineCancelled("使用者已取消處理")
        if self.cancel_event.is_set():
            raise PipelineCancelled("使用者已取消處理")

    def _fingerprint(self, item: MediaItem) -> str:
        transcript = self.transcript_dir / f"{item.item_id}.jsonl"
        stat = transcript.stat()
        reference_state: list[str] = []
        for raw_path in self.config.reference_sources or []:
            path = Path(raw_path)
            try:
                stat = path.stat()
                reference_state.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                reference_state.append(str(path))
        value = "|".join(
            [
                str(stat.st_size),
                str(stat.st_mtime_ns),
                ANALYSIS_CACHE_VERSION,
                self.config.whisper_model,
                self.config.transcription_source,
                self.config.transcription_mode,
                str(self.config.analysis_chunk_seconds),
                str(self.config.analysis_chunk_chars),
                self.config.analysis_scan_mode,
                self.config.output_language,
                self.config.llm_endpoint,
                self.config.llm_model,
                str(self.config.use_llm),
                "|".join(reference_state),
            ]
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _cache_path(self, item: MediaItem) -> Path:
        return self.cache_dir / f"{item.item_id}.analysis.json"

    def _load_analysis_cache(self, item: MediaItem) -> list[ExamCue] | None:
        path = self._cache_path(item)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != self._fingerprint(item):
                return None
            cues: list[ExamCue] = []
            for record in payload.get("cues", []):
                record["kind"] = CueKind(record["kind"])
                cues.append(ExamCue(**record))
            return cues
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _save_analysis_cache(self, item: MediaItem, cues: list[ExamCue]) -> None:
        path = self._cache_path(item)
        temp = path.with_suffix(".tmp")
        payload = {
            "fingerprint": self._fingerprint(item),
            "cues": [cue.to_dict() for cue in cues],
        }
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
