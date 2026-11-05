from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from examcue.analysis import CueAnalyzer, _analysis_ready_segments
from examcue.domain import MediaItem, TranscriptSegment
from examcue.util import atomic_write_json

BASELINE = Path("projects/fulltest")
MATERIALS = Path("projects/fulltest_materials")
DOMAIN_TERMS = (
    "opencv",
    "convolution",
    "gaussian",
    "filter",
    "kernel",
    "edge",
    "image",
    "pixel",
    "rgb",
    "bmp",
    "header",
    "hough",
    "transform",
    "homography",
    "affine",
    "matrix",
)


def load_segments(project: Path, item_id: str) -> list[TranscriptSegment]:
    path = project / "transcripts" / f"{item_id}.jsonl"
    segments: list[TranscriptSegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "meta" in payload:
            continue
        segments.append(
            TranscriptSegment(
                start=float(payload["start"]),
                end=float(payload["end"]),
                text=str(payload["text"]),
                speaker=str(payload.get("speaker") or "teacher"),
            )
        )
    return segments


def term_counts(segments: list[TranscriptSegment]) -> Counter[str]:
    text = " ".join(segment.text for segment in segments).casefold()
    return Counter({term: len(re.findall(re.escape(term), text)) for term in DOMAIN_TERMS})


def rule_cues(item: MediaItem, segments: list[TranscriptSegment]) -> list:
    analyzer = CueAnalyzer(use_llm=False, scan_mode="focused")
    return analyzer.analyze_item(item, segments)


def main() -> None:
    baseline_metrics = json.loads(
        (BASELINE / "run_metrics.json").read_text(encoding="utf-8")
    )
    material_metrics = json.loads(
        (MATERIALS / "run_metrics.json").read_text(encoding="utf-8")
    )
    rows: list[dict] = []
    old_rule_total = 0
    new_rule_total = 0
    examples: list[dict] = []

    for source in baseline_metrics["sources"]:
        item_id = source["item_id"]
        old = load_segments(BASELINE, item_id)
        new = load_segments(MATERIALS, item_id)
        item = MediaItem(source["source"], item_id, source["title"], source["source_type"])
        old_cues = rule_cues(item, old)
        new_cues = rule_cues(item, new)
        old_rule_total += len(old_cues)
        new_rule_total += len(new_cues)
        old_terms = term_counts(old)
        new_terms = term_counts(new)
        changed = 0
        similarity_sum = 0.0
        compared_chars = 0
        for before, after in zip(old, new, strict=False):
            before_text = before.text.strip()
            after_text = after.text.strip()
            weight = max(len(before_text), len(after_text), 1)
            if before_text != after_text:
                changed += 1
            similarity_sum += SequenceMatcher(
                None, before_text.casefold(), after_text.casefold(), autojunk=False
            ).ratio() * weight
            compared_chars += weight
            term_delta = sum(
                after_text.casefold().count(term) - before_text.casefold().count(term)
                for term in DOMAIN_TERMS
            )
            if term_delta > 0:
                examples.append(
                    {
                        "title": source["title"],
                        "start": after.start,
                        "before": before_text,
                        "after": after_text,
                        "term_delta": term_delta,
                    }
                )
        old_filtered = len(old) - len(_analysis_ready_segments(old))
        new_filtered = len(new) - len(_analysis_ready_segments(new))
        rows.append(
            {
                "item_id": item_id,
                "title": source["title"],
                "segments": len(new),
                "changed_segments": changed,
                "mean_text_similarity": round(
                    similarity_sum / max(compared_chars, 1), 4
                ),
                "old_characters": sum(len(value.text) for value in old),
                "new_characters": sum(len(value.text) for value in new),
                "old_domain_term_hits": sum(old_terms.values()),
                "new_domain_term_hits": sum(new_terms.values()),
                "domain_term_delta": sum(new_terms.values()) - sum(old_terms.values()),
                "old_filtered_segments": old_filtered,
                "new_filtered_segments": new_filtered,
                "old_rule_cues": len(old_cues),
                "new_rule_cues": len(new_cues),
            }
        )

    payload = {
        "baseline_project": str(BASELINE),
        "materials_project": str(MATERIALS),
        "baseline_llm_cues": baseline_metrics["cue_count"],
        "baseline_llm_model": baseline_metrics["llm"]["model"],
        "materials_reference_documents": material_metrics.get(
            "reference_document_count", 0
        ),
        "old_rule_cues": old_rule_total,
        "new_rule_cues": new_rule_total,
        "old_filtered_segments": sum(row["old_filtered_segments"] for row in rows),
        "new_filtered_segments": sum(row["new_filtered_segments"] for row in rows),
        "old_domain_term_hits": sum(row["old_domain_term_hits"] for row in rows),
        "new_domain_term_hits": sum(row["new_domain_term_hits"] for row in rows),
        "changed_segments": sum(row["changed_segments"] for row in rows),
        "total_segments": sum(row["segments"] for row in rows),
        "sources": rows,
        "top_term_improvement_examples": sorted(
            examples, key=lambda value: value["term_delta"], reverse=True
        )[:12],
    }
    atomic_write_json(MATERIALS / "comparison.json", payload)

    lines = [
        "# fulltest 教材 A/B 比較",
        "",
        f"- 逐字稿異動：{payload['changed_segments']}/{payload['total_segments']} 個片段",
        f"- 教材術語命中：{payload['old_domain_term_hits']} → {payload['new_domain_term_hits']}",
        f"- 疑似重複片段：{payload['old_filtered_segments']} → {payload['new_filtered_segments']}",
        f"- 同一規則分析考點：{old_rule_total} → {new_rule_total}",
        "",
        "| 影片 | 變更片段 | 文字相似度 | 術語命中 | 重複片段 | 規則考點 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['title']} | {row['changed_segments']}/{row['segments']} | "
            f"{row['mean_text_similarity']:.1%} | "
            f"{row['old_domain_term_hits']} → {row['new_domain_term_hits']} | "
            f"{row['old_filtered_segments']} → {row['new_filtered_segments']} | "
            f"{row['old_rule_cues']} → {row['new_rule_cues']} |"
        )
    lines.extend(["", "## 術語改善範例", ""])
    for example in payload["top_term_improvement_examples"]:
        lines.extend(
            [
                f"### {example['title']} @ {example['start']:.1f}s",
                "",
                f"- 原文：{example['before']}",
                f"- 教材版：{example['after']}",
                "",
            ]
        )
    (MATERIALS / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
