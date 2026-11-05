import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from examcue.analysis import (
    SYSTEM_PROMPT,
    CueAnalyzer,
    LLMStatus,
    OpenAICompatibleClient,
    _extract_json_array,
    _is_severely_repetitive,
    chunk_has_exam_signal,
    chunk_segments,
    focused_chunk_segments,
    heuristic_analyze,
)
from examcue.domain import CueKind, MediaItem, TranscriptSegment


class HeuristicAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.item = MediaItem("lecture.mp4", "a", "Lecture", "file")

    def test_classifies_explicit_exam_cue(self):
        segments = [
            TranscriptSegment(0, 2, "This is the camera model."),
            TranscriptSegment(2, 5, "I will ask you this on the exam."),
            TranscriptSegment(5, 8, "The answer is focal length."),
        ]
        cues = heuristic_analyze(self.item, segments)
        self.assertEqual(cues[0].kind, CueKind.EXPLICIT)
        self.assertIn("exam", cues[0].evidence)

    def test_classifies_exclusion_before_generic_exam_words(self):
        segments = [TranscriptSegment(0, 3, "You do not need to prove this equation.")]
        cues = heuristic_analyze(self.item, segments)
        self.assertEqual(cues[0].kind, CueKind.EXCLUSION)

    def test_classifies_warning_and_inference(self):
        segments = [
            TranscriptSegment(0, 3, "Be careful with the units."),
            TranscriptSegment(4, 7, "The focal length is very important."),
        ]
        cues = heuristic_analyze(self.item, segments)
        self.assertEqual([cue.kind for cue in cues], [CueKind.WARNING, CueKind.INFERRED])

    def test_conservative_fallback_does_not_infer_from_emphasis(self):
        segments = [
            TranscriptSegment(0, 3, "但是重點不在這裡"),
            TranscriptSegment(3, 6, "這個考試會考"),
        ]
        cues = heuristic_analyze(
            self.item,
            segments,
            include_inferred=False,
            analysis_method="fallback",
        )

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].kind, CueKind.EXPLICIT)
        self.assertEqual(cues[0].analysis_method, "fallback")

    def test_fallback_rejects_exam_analogies_and_old_exam_stories(self):
        segments = [
            TranscriptSegment(0, 10, "大家都是像考試一樣都考20幾分，老師要檢討。"),
            TranscriptSegment(10, 20, "他去瑞昱考試然後考完才知道只有十幾分。"),
            TranscriptSegment(20, 30, "現在幾分，我們剛剛提到16分。"),
        ]

        cues = heuristic_analyze(
            self.item,
            segments,
            include_inferred=False,
            analysis_method="fallback",
        )

        self.assertEqual(cues, [])

    def test_fallback_uses_local_signal_quote_instead_of_long_segment(self):
        text = (
            "前面是一般介紹，這裡要注意一下旋轉矩陣的X軸不會改變，後面繼續講很多與提醒無關的內容。"
        )

        cues = heuristic_analyze(
            self.item,
            [TranscriptSegment(100, 140, text)],
            include_inferred=False,
            analysis_method="fallback",
        )

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].kind, CueKind.WARNING)
        self.assertLessEqual(len(cues[0].title), 46)
        self.assertLessEqual(len(cues[0].evidence), 96)
        self.assertEqual(cues[0].summary, cues[0].evidence)
        self.assertEqual((cues[0].start, cues[0].end), (100, 140))

    def test_default_chunks_are_limited_to_two_minutes(self):
        segments = [
            TranscriptSegment(0, 60, "first"),
            TranscriptSegment(60, 121, "second"),
            TranscriptSegment(121, 180, "third"),
        ]

        chunks = list(chunk_segments(segments))

        self.assertEqual(len(chunks), 2)

    def test_analyzer_uses_selected_chunk_limits(self):
        analyzer = CueAnalyzer(
            use_llm=False,
            chunk_max_seconds=60,
            chunk_max_chars=2000,
        )
        segments = [
            TranscriptSegment(0, 40, "first"),
            TranscriptSegment(40, 80, "second"),
            TranscriptSegment(80, 120, "third"),
        ]

        with patch("examcue.analysis.chunk_segments", wraps=chunk_segments) as chunks:
            analyzer.analyze_item(self.item, segments)

        self.assertEqual(chunks.call_args.kwargs["max_seconds"], 60)
        self.assertEqual(chunks.call_args.kwargs["max_chars"], 2000)

    def test_candidate_filter_rejects_neutral_content(self):
        neutral = [TranscriptSegment(0, 10, "今天介紹影像座標系統。")]
        important = [TranscriptSegment(0, 10, "這個差別非常重要，考試會考。")]

        self.assertFalse(chunk_has_exam_signal(neutral))
        self.assertTrue(chunk_has_exam_signal(important))

    def test_focused_chunks_keep_context_and_drop_distant_neutral_content(self):
        segments = [
            TranscriptSegment(index * 10, index * 10 + 10, f"一般內容 {index}")
            for index in range(12)
        ]
        segments[5].text = "這裡非常重要，考試會考"

        chunks = list(focused_chunk_segments(segments, max_seconds=120, max_chars=4000))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0].text, "一般內容 3")
        self.assertEqual(chunks[0][-1].text, "一般內容 9")

    def test_focused_chunks_detect_signal_split_across_segments(self):
        segments = [
            TranscriptSegment(0, 5, "這次考試會"),
            TranscriptSegment(5, 10, "考到影像座標"),
        ]

        chunks = list(focused_chunk_segments(segments))

        self.assertEqual(chunks, [segments])

    def test_focused_chunks_pack_distant_windows_by_content_limits(self):
        segments = [
            TranscriptSegment(0, 5, "考試會考第一個概念"),
            TranscriptSegment(5, 10, "第一個概念的答案"),
            TranscriptSegment(300, 305, "中間一般內容"),
            TranscriptSegment(600, 605, "小心第二個常見錯誤"),
        ]

        chunks = list(
            focused_chunk_segments(
                segments,
                max_seconds=120,
                max_chars=4000,
                context_before=0,
                context_after=0,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual([segment.start for segment in chunks[0]], [0, 600])

    def test_focused_chunks_limit_independent_windows_per_request(self):
        segments = [
            TranscriptSegment(index * 100, index * 100 + 5, f"第 {index} 個考試會考")
            for index in range(4)
        ]

        chunks = list(
            focused_chunk_segments(
                segments,
                max_seconds=120,
                max_chars=4000,
                context_before=0,
                context_after=0,
                max_windows_per_chunk=3,
            )
        )

        self.assertEqual([len(chunk) for chunk in chunks], [3, 1])

    def test_parser_accepts_single_cue_object(self):
        record = {
            "kind": "explicit",
            "title": "考點",
            "segment_ids": [0],
        }

        self.assertEqual(_extract_json_array(json.dumps(record)), [record])

    def test_parser_and_converter_accept_compact_cue_keys(self):
        record = {
            "k": "explicit",
            "t": "DPI 定義",
            "e": "這個考試會考",
            "b": "direct_exam",
            "c": 0.96,
            "s": [0],
        }
        analyzer = CueAnalyzer(use_llm=False)

        parsed = _extract_json_array(json.dumps(record))
        cues = analyzer._records_to_cues(
            parsed,
            self.item,
            [TranscriptSegment(0, 10, "這個考試會考 DPI 定義")],
        )

        self.assertEqual(cues[0].kind, CueKind.EXPLICIT)
        self.assertEqual(cues[0].title, "DPI 定義")
        self.assertEqual(cues[0].confidence, 0.96)

    def test_parser_recovers_complete_objects_from_truncated_array(self):
        content = '[{"kind":"explicit","title":"考點","segment_ids":[0]}, {"kind"'

        records = _extract_json_array(content)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "考點")

    def test_successful_llm_chunks_are_reused_after_interruption(self):
        records = [
            {
                "kind": "explicit",
                "title": "明確考點",
                "evidence": "考試會考",
                "reason": "老師直接說明",
                "confidence": 0.95,
                "segment_ids": [0],
            }
        ]
        segments = [TranscriptSegment(0, 10, "這個考試會考")]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            analyzer = CueAnalyzer(
                use_llm=True,
                endpoint="http://localhost:1234/v1",
                model="test-model",
                chunk_cache_dir=Path(directory),
            )
            analyzer.client.status = LLMStatus(analyzer.client.requested_endpoint, "test-model")
            analyzer.client.analyze = Mock(return_value=records)

            first = analyzer.analyze_item(self.item, segments)
            second = analyzer.analyze_item(self.item, segments)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        analyzer.client.analyze.assert_called_once()

    def test_empty_direct_signal_response_is_retried_instead_of_cached(self):
        segments = [TranscriptSegment(0, 10, "這個定義考試會考")]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            analyzer = CueAnalyzer(
                use_llm=True,
                endpoint="http://localhost:1234/v1",
                model="weak-model",
                chunk_cache_dir=Path(directory),
            )
            analyzer.client.status = LLMStatus(analyzer.client.requested_endpoint, "weak-model")
            analyzer.client.analyze = Mock(return_value=[])

            analyzer.analyze_item(self.item, segments)
            analyzer.analyze_item(self.item, segments)

            self.assertEqual(list(Path(directory).glob("*.json")), [])

        self.assertEqual(analyzer.client.analyze.call_count, 2)

    def test_cache_write_failure_keeps_successful_llm_result(self):
        records = [
            {
                "k": "explicit",
                "t": "明確考點",
                "e": "考試會考",
                "b": "direct_exam",
                "c": 0.95,
                "s": [0],
            }
        ]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            analyzer = CueAnalyzer(
                use_llm=True,
                endpoint="http://localhost:1234/v1",
                model="test-model",
                chunk_cache_dir=Path(directory),
            )
            analyzer.client.status = LLMStatus(analyzer.client.requested_endpoint, "test-model")
            analyzer.client.analyze = Mock(return_value=records)
            with patch(
                "examcue.analysis.atomic_write_json",
                side_effect=OSError("disk busy"),
            ):
                cues = analyzer.analyze_item(
                    self.item,
                    [TranscriptSegment(0, 10, "這個考試會考")],
                )

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].analysis_method, "llm")
        self.assertEqual(analyzer._llm_failure_streak, 0)
        self.assertTrue(any("快取寫入失敗" in warning for warning in analyzer.warnings))

    def test_full_scan_sends_neutral_chunks_to_llm(self):
        analyzer = CueAnalyzer(
            use_llm=True,
            endpoint="http://localhost:1234/v1",
            model="test-model",
            scan_mode="full",
        )
        analyzer.client.status = LLMStatus(analyzer.client.requested_endpoint, "test-model")
        analyzer.client.analyze = Mock(return_value=[])

        analyzer.analyze_item(
            self.item,
            [TranscriptSegment(0, 10, "今天介紹影像座標系統。")],
        )

        analyzer.client.analyze.assert_called_once()

    def test_focused_scan_with_no_candidates_does_not_connect(self):
        analyzer = CueAnalyzer(use_llm=True, scan_mode="focused")
        analyzer.client.connect = Mock(
            side_effect=AssertionError("neutral transcript must not contact the LLM")
        )

        cues = analyzer.analyze_item(
            self.item,
            [TranscriptSegment(0, 10, "今天介紹影像座標系統。")],
        )

        self.assertEqual(cues, [])
        analyzer.client.connect.assert_not_called()

    def test_compact_basis_is_expanded_into_readable_reason(self):
        analyzer = CueAnalyzer(use_llm=False)
        records = [
            {
                "kind": "warning",
                "title": "不要混淆兩種座標",
                "evidence": "這兩個不要搞混",
                "basis": "warning",
                "confidence": 0.9,
                "segment_ids": [0],
            }
        ]

        cues = analyzer._records_to_cues(
            records,
            self.item,
            [TranscriptSegment(0, 10, "這兩個不要搞混")],
        )

        self.assertEqual(cues[0].reason, "老師直接提醒常見錯誤或容易混淆之處")

    def test_important_without_exam_statement_is_downgraded_to_inferred(self):
        analyzer = CueAnalyzer(use_llm=False)
        cues = analyzer._records_to_cues(
            [
                {
                    "k": "explicit",
                    "t": "Image Format",
                    "e": "Image Format比較重要",
                    "b": "direct_exam",
                    "c": 0.95,
                    "s": [0],
                }
            ],
            self.item,
            [TranscriptSegment(0, 10, "這個Image Format比較重要")],
        )

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].kind, CueKind.INFERRED)
        self.assertEqual(cues[0].reason, "推論依據：老師明確強調這項內容的重要性")
        self.assertLessEqual(cues[0].confidence, 0.88)

    def test_not_discussed_is_not_accepted_as_exam_exclusion(self):
        analyzer = CueAnalyzer(use_llm=False)
        cues = analyzer._records_to_cues(
            [
                {
                    "k": "exclusion",
                    "t": "Hough Transform",
                    "e": "Hough Transform我就不講了",
                    "b": "exclusion",
                    "c": 0.9,
                    "s": [0],
                }
            ],
            self.item,
            [TranscriptSegment(0, 10, "Hough Transform我就不講了")],
        )

        self.assertEqual(cues, [])
        self.assertEqual(analyzer.quality_rejected_cue_count, 1)

    def test_ordinary_canonical_answer_is_not_enough_for_inference(self):
        analyzer = CueAnalyzer(use_llm=False)
        records = [
            {
                "k": "inferred",
                "t": "AXB矩陣",
                "e": "A就是Matrix",
                "b": "canonical_answer",
                "c": 0.8,
                "s": [0],
            }
        ]

        cues = analyzer._records_to_cues(
            records,
            self.item,
            [TranscriptSegment(0, 10, "A就是Matrix，X是Input")],
        )

        self.assertEqual(cues, [])

    def test_canonical_answer_near_direct_exam_cue_can_be_inferred(self):
        analyzer = CueAnalyzer(use_llm=False)
        records = [
            {
                "k": "inferred",
                "t": "Norm與高斯模型",
                "e": "Norm就是高斯模型的指數部分",
                "b": "canonical_answer",
                "c": 0.85,
                "s": [1],
            }
        ]
        segments = [
            TranscriptSegment(0, 10, "考試會考Norm是相似度量測"),
            TranscriptSegment(10, 20, "Norm就是高斯模型的指數部分"),
        ]

        cues = analyzer._records_to_cues(records, self.item, segments)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].kind, CueKind.INFERRED)

    def test_repetitive_asr_loop_is_detected(self):
        text = "我們接著講 Perspective " + "Perspective " * 20

        self.assertTrue(_is_severely_repetitive(text))
        self.assertFalse(_is_severely_repetitive("這個概念很重要，所以再說明一次"))

    def test_llm_evidence_must_be_an_exact_quote(self):
        analyzer = CueAnalyzer(use_llm=False)
        records = [
            {
                "k": "explicit",
                "t": "DPI",
                "e": "老師說DPI一定會考",
                "b": "direct_exam",
                "c": 0.95,
                "s": [0],
            }
        ]

        cues = analyzer._records_to_cues(
            records,
            self.item,
            [TranscriptSegment(0, 10, "這個DPI考試會考")],
        )

        self.assertEqual(cues, [])

    def test_llm_ellipsis_evidence_is_rebuilt_from_selected_segments(self):
        analyzer = CueAnalyzer(use_llm=False)
        records = [
            {
                "k": "inferred",
                "t": "Focal Length 與 Principal Point",
                "e": ("最後一個是Principal Point這個很重要... Focal Length是最重要的所以你要知道"),
                "b": "emphasis",
                "c": 0.9,
                "s": [0, 1],
            }
        ]
        segments = [
            # The model omitted one spoken filler word ("叫") around its quote.
            TranscriptSegment(0, 10, "最後一個是叫Principal Point這個很重要"),
            TranscriptSegment(10, 20, "Focal Length是最重要的所以你要知道"),
        ]

        cues = analyzer._records_to_cues(records, self.item, segments)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].kind, CueKind.INFERRED)
        self.assertNotIn("...", cues[0].evidence)
        self.assertIn(cues[0].evidence, " ".join(segment.text for segment in segments))

    def test_prompt_repeats_precision_rules_near_transcript(self):
        analyzer = CueAnalyzer(use_llm=False)
        prompt = analyzer._prompt_for_chunk(
            self.item,
            [TranscriptSegment(0, 10, "Image Format比較重要")],
        )

        self.assertIn("Important without a direct exam statement is inferred", prompt)
        self.assertIn("Not discussed is not the same as not tested", prompt)

    def test_disables_llm_after_two_consecutive_chunk_failures(self):
        analyzer = CueAnalyzer(
            use_llm=True,
            endpoint="http://localhost:1234/v1",
            model="test-model",
            scan_mode="full",
            chunk_max_seconds=10,
        )
        analyzer.client.status = LLMStatus(analyzer.client.requested_endpoint, "test-model")
        analyzer.client.analyze = Mock(side_effect=TimeoutError("timed out"))
        segments = [
            TranscriptSegment(0, 10, "第一段考試會考"),
            TranscriptSegment(11, 20, "第二段考試會考"),
            TranscriptSegment(21, 30, "第三段考試會考"),
        ]

        analyzer.analyze_item(self.item, segments)

        self.assertEqual(analyzer.client.analyze.call_count, 2)
        self.assertTrue(analyzer._llm_disabled)
        self.assertFalse(any("無法連線至指定" in value for value in analyzer.warnings))

    def test_empty_llm_results_on_direct_cues_use_fallback_without_stopping_llm(self):
        analyzer = CueAnalyzer(
            use_llm=True,
            endpoint="http://localhost:1234/v1",
            model="weak-model",
            scan_mode="full",
            chunk_max_seconds=10,
        )
        analyzer.client.status = LLMStatus(analyzer.client.requested_endpoint, "weak-model")
        analyzer.client.analyze = Mock(return_value=[])
        segments = [
            TranscriptSegment(0, 10, "第一個定義考試會考"),
            TranscriptSegment(11, 20, "第二個公式考試不會考"),
            TranscriptSegment(21, 30, "第三個觀念考試會考"),
        ]

        cues = analyzer.analyze_item(self.item, segments)

        self.assertEqual(analyzer.client.analyze.call_count, 3)
        self.assertEqual(analyzer.empty_direct_response_count, 3)
        self.assertFalse(analyzer._llm_disabled)
        self.assertTrue(cues)
        self.assertTrue(all(cue.analysis_method == "fallback" for cue in cues))
        self.assertTrue(any("仍會繼續" in value for value in analyzer.warnings))


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_lm_studio_vision_uses_native_api_with_reasoning_off(self):
        client = OpenAICompatibleClient(
            endpoint="http://localhost:1234/v1", model="qwen/qwen3.5-9b"
        )
        client.status = LLMStatus(endpoint=client.requested_endpoint, model=client.requested_model)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            image = Path(directory) / "frame.jpg"
            image.write_bytes(b"jpeg")
            with patch.object(
                client,
                "_request_json",
                return_value={"output": [{"type": "message", "content": '[{"i":0,"d":"slide"}]'}]},
            ) as request:
                records = client.analyze_images([image], "Describe")

        self.assertEqual(records[0]["d"], "slide")
        self.assertTrue(request.call_args.args[0].endswith("/api/v1/chat"))
        payload = request.call_args.kwargs["payload"]
        self.assertEqual(payload["reasoning"], "off")
        self.assertEqual(payload["input"][1]["type"], "image")

    def test_analysis_request_allows_slow_local_model(self):
        client = OpenAICompatibleClient(
            endpoint="http://localhost:1234/v1", model="qwen/qwen3.8-27b"
        )
        client.status = LLMStatus(endpoint=client.requested_endpoint, model="qwen")
        with patch.object(
            client,
            "_request_json",
            return_value={"output": [{"type": "message", "content": "[]"}]},
        ) as request:
            self.assertEqual(client.analyze("test"), [])

        self.assertEqual(request.call_args.kwargs["timeout"], 300)
        self.assertTrue(request.call_args.args[0].endswith("/api/v1/chat"))
        payload = request.call_args.kwargs["payload"]
        self.assertEqual(payload["max_output_tokens"], 512)
        self.assertEqual(payload["reasoning"], "off")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["input"], "test")
        self.assertEqual(payload["system_prompt"], SYSTEM_PROMPT)

    def test_lm_studio_native_response_without_message_is_rejected(self):
        client = OpenAICompatibleClient(
            endpoint="http://localhost:1234/v1", model="qwen/qwen3.8-27b"
        )
        client.status = LLMStatus(endpoint=client.requested_endpoint, model=client.requested_model)
        with (
            patch.object(
                client,
                "_request_json",
                return_value={"output": [{"type": "reasoning", "content": "thinking"}]},
            ),
            self.assertRaisesRegex(ValueError, "只回傳推理內容"),
        ):
            client.analyze("test")

    def test_external_model_does_not_receive_qwen_specific_options(self):
        client = OpenAICompatibleClient(endpoint="https://example.test/v1", model="example-model")
        client.status = LLMStatus(endpoint=client.requested_endpoint, model="example-model")
        with patch.object(
            client,
            "_request_json",
            return_value={"choices": [{"message": {"content": "[]"}}]},
        ) as request:
            client.analyze("test")

        payload = request.call_args.kwargs["payload"]
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertFalse(payload["messages"][-1]["content"].endswith("/no_think"))

    def test_reuses_successful_model_discovery(self):
        client = OpenAICompatibleClient(endpoint="https://example.test/v1")
        with patch.object(
            client,
            "_request_json",
            return_value={"data": [{"id": "example-model"}]},
        ) as request:
            first = client.connect()
            second = client.connect()

        self.assertIs(first, second)
        request.assert_called_once()

    def test_does_not_repeat_failed_model_discovery_for_every_video(self):
        client = OpenAICompatibleClient(endpoint="https://unavailable.test/v1")
        with patch.object(client, "_request_json", side_effect=OSError("offline")) as request:
            self.assertIsNone(client.connect())
            self.assertIsNone(client.connect())

        request.assert_called_once()

    def test_connection_checks_chat_completion_for_selected_model(self):
        client = OpenAICompatibleClient(
            endpoint="https://example.test/v1", model="example-model", api_key="secret-key"
        )
        with patch.object(
            client,
            "_request_json",
            side_effect=[
                {"data": [{"id": "example-model"}]},
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    [
                                        {
                                            "k": "explicit",
                                            "t": "Focal length definition",
                                            "e": "This definition will be on the exam.",
                                            "b": "direct_exam",
                                            "c": 0.95,
                                            "s": [0],
                                        }
                                    ],
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                },
            ],
        ) as request:
            status = client.test_connection()

        self.assertEqual(status.model, "example-model")
        self.assertEqual(request.call_count, 2)
        self.assertTrue(request.call_args_list[1].args[0].endswith("/chat/completions"))

    def test_connection_rejects_model_that_returns_empty_analysis(self):
        client = OpenAICompatibleClient(endpoint="https://example.test/v1", model="base-model")
        with (
            patch.object(
                client,
                "_request_json",
                side_effect=[
                    {"data": [{"id": "base-model"}]},
                    {"choices": [{"message": {"content": "[]"}}]},
                ],
            ),
            self.assertRaisesRegex(ValueError, "考點分析測試"),
        ):
            client.test_connection()

    def test_connection_rejects_reasoning_only_response(self):
        client = OpenAICompatibleClient(
            endpoint="http://localhost:11434/v1", model="qwen/qwen3.8-27b"
        )
        with (
            patch.object(
                client,
                "_request_json",
                side_effect=[
                    {"data": [{"id": "qwen/qwen3.8-27b"}]},
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "reasoning_content": "The user asked me to reply with OK",
                                },
                                "finish_reason": "length",
                            }
                        ]
                    },
                ],
            ),
            self.assertRaisesRegex(ValueError, "只回傳推理內容"),
        ):
            client.test_connection()

    def test_sends_api_key_as_bearer_header(self):
        client = OpenAICompatibleClient(
            endpoint="https://example.test/v1", model="example-model", api_key="secret-key"
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps({"data": [{"id": "example-model"}]}).encode()

        def fake_urlopen(request, timeout):
            self.assertEqual(request.get_header("Authorization"), "Bearer secret-key")
            self.assertEqual(timeout, 3)
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            status = client.connect()

        self.assertIsNotNone(status)
        self.assertEqual(status.model, "example-model")


if __name__ == "__main__":
    unittest.main()
