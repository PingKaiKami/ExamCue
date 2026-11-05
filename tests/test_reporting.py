import unittest

from examcue.domain import AnalysisReport, CueKind, ExamCue
from examcue.reporting import render_markdown


class ReportTests(unittest.TestCase):
    def test_youtube_timestamp_and_labels(self):
        cue = ExamCue(
            kind=CueKind.EXPLICIT,
            title="Camera parameters",
            summary="Know four parameters.",
            evidence="This is on the exam.",
            reason="Direct statement",
            confidence=0.95,
            source_id="v1",
            source_title="Lecture 1",
            start=125,
            end=130,
            analysis_method="llm",
        )
        report = AnalysisReport(
            project_name="Course",
            output_language="English",
            cues=[cue],
            sources=[
                {
                    "item_id": "v1",
                    "title": "Lecture 1",
                    "webpage_url": "https://youtube.com/watch?v=abc",
                    "transcription_method_label": "YouTube 自動字幕",
                    "transcript_segments": 12,
                }
            ],
        )
        markdown = render_markdown(report)
        self.assertIn("明示考點", markdown)
        self.assertIn("t=125", markdown)
        self.assertIn("02:05", markdown)
        self.assertIn("分析方式：LLM 判斷", markdown)
        self.assertIn("Lecture 1：YouTube 自動字幕（12 個片段）", markdown)


if __name__ == "__main__":
    unittest.main()
