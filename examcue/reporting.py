from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .domain import AnalysisReport, CueKind
from .util import seconds_to_timestamp

KIND_LABELS = {
    CueKind.EXPLICIT: "明示考點",
    CueKind.INFERRED: "推論考點",
    CueKind.WARNING: "易錯／警告",
    CueKind.EXCLUSION: "明確排除範圍",
}

ANALYSIS_METHOD_LABELS = {
    "llm": "LLM 判斷",
    "rules": "規則式分析",
    "fallback": "規則式備援",
    "unknown": "未記錄",
}


def write_report(report: AnalysisReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: AnalysisReport) -> str:
    lines = [
        f"# {report.project_name} — ExamCue 分析報告",
        "",
        (
            "> 「明示考點」具有老師直接提到考試／出題的證據；"
            "「推論考點」是系統根據強調、重複或標準答案推測，並不代表老師保證會考。"
        ),
        "",
    ]
    source_map = {
        str(source["item_id"]): source for source in report.sources if "item_id" in source
    }
    if report.sources:
        lines.extend(["## 轉錄來源", ""])
        for source in report.sources:
            title = str(source.get("title") or source.get("item_id") or "未命名來源")
            method = str(
                source.get("transcription_method_label")
                or source.get("transcription_method")
                or "未記錄"
            )
            segments = source.get("transcript_segments", 0)
            lines.append(f"- {title}：{method}（{segments} 個片段）")
        lines.append("")
    if report.reference_documents:
        lines.extend(["## 輔助教材", ""])
        for document in report.reference_documents:
            lines.append(
                f"- {document.get('name', '未命名教材')}"
                f"（擷取 {int(document.get('characters', 0)):,} 字元）"
            )
        lines.extend(
            [
                "",
                "> 教材僅用於術語校正與上下文理解，不會單獨作為老師明示考點的證據。",
                "",
            ]
        )
    if report.visual_observations:
        described = sum(bool(item.description) for item in report.visual_observations)
        lines.extend(
            [
                "## 課程畫面",
                "",
                (
                    f"已定時擷取 {len(report.visual_observations)} 張畫面；"
                    f"其中 {described} 張具有模型描述。原始圖片位於 `frames/`。"
                ),
                "",
            ]
        )
        for item in report.visual_observations:
            if item.description:
                lines.append(
                    f"- {item.source_title} {seconds_to_timestamp(item.timestamp)}："
                    f"{item.description}"
                )
        lines.append("")
    for kind in (
        CueKind.EXPLICIT,
        CueKind.INFERRED,
        CueKind.WARNING,
        CueKind.EXCLUSION,
    ):
        cues = [cue for cue in report.cues if cue.kind == kind]
        lines.extend([f"## {KIND_LABELS[kind]}（{len(cues)}）", ""])
        if not cues:
            lines.extend(["目前沒有找到。", ""])
            continue
        for index, cue in enumerate(
            sorted(cues, key=lambda value: (-value.confidence, value.source_title, value.start)),
            start=1,
        ):
            source = source_map.get(cue.source_id, {})
            timestamp = seconds_to_timestamp(cue.start)
            source_label = f"{cue.source_title} {timestamp}"
            source_url = _timestamp_url(str(source.get("webpage_url") or ""), cue.start)
            source_display = f"[{source_label}]({source_url})" if source_url else source_label
            lines.extend(
                [
                    f"### {index}. {cue.title}",
                    "",
                    cue.summary,
                    "",
                    f"- 分類：{KIND_LABELS[cue.kind]}",
                    f"- 分析方式：{ANALYSIS_METHOD_LABELS.get(cue.analysis_method, cue.analysis_method)}",
                    f"- 可信度：{cue.confidence:.0%}",
                    f"- 判定理由：{cue.reason or '未提供'}",
                    f"- 證據：「{cue.evidence}」",
                    f"- 來源：{source_display}",
                ]
            )
            if len(cue.occurrences) > 1:
                lines.append(f"- 相似內容出現：{len(cue.occurrences)} 次")
            lines.append("")

    if report.warnings:
        lines.extend(["## 處理警告", ""])
        lines.extend(f"- {warning}" for warning in dict.fromkeys(report.warnings))
        lines.append("")
    if report.quiz_questions:
        lines.extend(
            [
                "## 練習題",
                "",
                f"已依考點產生 {len(report.quiz_questions)} 題四選一題目，詳見 `quiz.md`。",
                "",
            ]
        )
    return "\n".join(lines)


def _timestamp_url(url: str, seconds: float) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["t"] = [str(max(0, int(seconds)))]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
