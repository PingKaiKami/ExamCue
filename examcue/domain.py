from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class CueKind(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    EXCLUSION = "exclusion"
    WARNING = "warning"


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "teacher"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MediaItem:
    source: str
    item_id: str
    title: str
    source_type: str
    audio_path: Path | None = None
    subtitle_path: Path | None = None
    subtitle_kind: str | None = None
    webpage_url: str | None = None
    video_path: Path | None = None


@dataclass(slots=True)
class VisualObservation:
    source_id: str
    source_title: str
    timestamp: float
    image_path: Path
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return data


@dataclass(slots=True)
class QuizQuestion:
    question: str
    options: list[str]
    answer_index: int
    explanation: str
    source_id: str
    source_title: str
    start: float
    cue_title: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExamCue:
    kind: CueKind
    title: str
    summary: str
    evidence: str
    reason: str
    confidence: float
    source_id: str
    source_title: str
    start: float
    end: float
    original_language: str | None = None
    topic: str | None = None
    analysis_method: str = "unknown"
    occurrences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(slots=True)
class AnalysisReport:
    project_name: str
    output_language: str
    cues: list[ExamCue]
    sources: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    visual_observations: list[VisualObservation] = field(default_factory=list)
    reference_documents: list[dict[str, Any]] = field(default_factory=list)
    quiz_questions: list[QuizQuestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "output_language": self.output_language,
            "cues": [cue.to_dict() for cue in self.cues],
            "sources": self.sources,
            "warnings": self.warnings,
            "visual_observations": [item.to_dict() for item in self.visual_observations],
            "reference_documents": self.reference_documents,
            "quiz_questions": [question.to_dict() for question in self.quiz_questions],
        }
