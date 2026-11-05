from pathlib import Path
from unittest.mock import patch

from examcue.analysis import CueAnalyzer, _relevant_reference_context
from examcue.domain import CueKind, ExamCue, MediaItem, TranscriptSegment, VisualObservation
from examcue.quiz import generate_quiz
from examcue.references import load_reference_corpus, retrieve_reference_context
from examcue.vision import capture_periodic_frames, describe_frames


def _cue() -> ExamCue:
    return ExamCue(
        kind=CueKind.EXPLICIT,
        title="Focal length",
        summary="Focal length controls field of view.",
        evidence="This will be on the exam: focal length controls field of view.",
        reason="direct exam statement",
        confidence=0.95,
        source_id="lesson-1",
        source_title="Lesson 1",
        start=12.0,
        end=18.0,
    )


def test_reference_text_builds_context_and_hotwords(tmp_path: Path) -> None:
    source = tmp_path / "slides.txt"
    source.write_text("Homography homography\nProjective geometry\n焦距 焦距", encoding="utf-8")

    corpus = load_reference_corpus([str(source)])

    assert corpus.documents[0]["name"] == "slides.txt"
    assert "Projective geometry" in corpus.context
    assert "Homography" in corpus.hotwords


def test_reference_corpus_keeps_later_documents_beyond_old_limit(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("A" * 13000, encoding="utf-8")
    second.write_text("LaterDocumentTerm", encoding="utf-8")

    corpus = load_reference_corpus([str(first), str(second)])

    assert "LaterDocumentTerm" in corpus.context


def test_reference_retrieval_prefers_matching_paragraph() -> None:
    context = "Color theory and pigments.\n\nHomography maps projective planes."
    result = _relevant_reference_context(context, "Today we derive a homography", limit=40)
    assert "Homography" in result


def test_reference_index_is_reused_until_a_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "slides.txt"
    cache = tmp_path / "reference-index.json"
    source.write_text("Homography maps projective planes.", encoding="utf-8")

    first = load_reference_corpus([str(source)], cache_path=cache)
    with patch(
        "examcue.references.extract_reference_text",
        side_effect=AssertionError("material should be loaded from its index"),
    ):
        second = load_reference_corpus([str(source)], cache_path=cache)

    assert first.chunks
    assert second.chunks == first.chunks
    assert cache.exists()


def test_short_reference_retrieval_omits_unrelated_material(tmp_path: Path) -> None:
    source = tmp_path / "slides.txt"
    source.write_text(
        "Color theory and pigments.\n\nHomography maps projective planes.",
        encoding="utf-8",
    )
    corpus = load_reference_corpus([str(source)])

    matching = retrieve_reference_context(corpus.chunks, "Today we derive a homography", limit=120)
    unrelated = retrieve_reference_context(
        corpus.chunks, "The teacher announces next week's office hours", limit=120
    )

    assert "Homography" in matching
    assert len(matching) <= 120
    assert unrelated == ""


def test_reference_retrieval_supports_non_latin_terms(tmp_path: Path) -> None:
    source = tmp_path / "lesson.txt"
    source.write_text(
        "Проективная геометрия использует гомографию.\n\n"
        "الهندسة الإسقاطية تستخدم التحويل التجانسي.",
        encoding="utf-8",
    )
    corpus = load_reference_corpus([str(source)])

    assert "Проективная" in retrieve_reference_context(
        corpus.chunks, "Объясните проективную геометрию", limit=200
    )
    assert "الهندسة" in retrieve_reference_context(
        corpus.chunks, "ما هي الهندسة الإسقاطية؟", limit=200
    )


def test_analyzer_sends_only_short_relevant_material(tmp_path: Path) -> None:
    source = tmp_path / "slides.txt"
    source.write_text(
        "Color theory and pigments.\n\nHomography maps projective planes.",
        encoding="utf-8",
    )
    corpus = load_reference_corpus([str(source)])
    analyzer = CueAnalyzer(
        use_llm=False,
        reference_chunks=corpus.chunks,
        reference_prompt_chars=120,
    )

    prompt = analyzer._prompt_for_chunk(
        MediaItem("lesson.mp3", "lesson", "Lesson", "file"),
        [TranscriptSegment(0, 10, "Today we derive a homography")],
    )

    reference = prompt.split("Reference material", 1)[1].split("Transcript segments", 1)[0]
    assert "Homography" in reference
    assert len(reference) < 180


def test_frame_capture_is_empty_without_video(tmp_path: Path) -> None:
    item = MediaItem("audio.mp3", "id", "Audio", "file")
    assert capture_periodic_frames(item, tmp_path) == []


def test_quiz_generation_keeps_source_metadata() -> None:
    class FakeClient:
        def complete_json(self, prompt: str, system: str, max_tokens: int) -> list[dict]:
            assert "Focal length" in prompt
            return [
                {
                    "q": "What controls field of view?",
                    "o": ["Focal length", "ISO", "Codec", "Bitrate"],
                    "a": 0,
                    "e": "The verified cue states this relationship.",
                    "i": 0,
                }
            ]

    questions = generate_quiz([_cue()], FakeClient(), "English", 10)  # type: ignore[arg-type]
    assert questions[0].answer_index == 0
    assert questions[0].source_title == "Lesson 1"


def test_quiz_generation_isolates_each_cue() -> None:
    prompts: list[str] = []

    class FakeClient:
        def complete_json(self, prompt: str, system: str, max_tokens: int) -> list[dict]:
            prompts.append(prompt)
            title = "Focal length" if "Focal length" in prompt else "ISO"
            return [
                {
                    "q": f"Question about {title}?",
                    "o": [title, "B", "C", "D"],
                    "a": 0,
                    "e": "Grounded in this cue.",
                    "i": 0,
                }
            ]

    second = _cue()
    second.title = "ISO"
    second.summary = "ISO controls sensor sensitivity."
    second.evidence = "This will be on the exam: ISO controls sensor sensitivity."
    questions = generate_quiz([_cue(), second], FakeClient(), "English", 10)  # type: ignore[arg-type]

    assert len(questions) == 2
    assert len(prompts) == 2
    assert "ISO" not in prompts[0]
    assert "Focal length" not in prompts[1]


def test_visual_description_deduplicates_identical_frames(tmp_path: Path) -> None:
    from PIL import Image

    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    for path in paths:
        Image.new("RGB", (32, 32), "white").save(path)
    observations = [
        VisualObservation("id", "Lecture", float(index), path) for index, path in enumerate(paths)
    ]

    class FakeClient:
        calls = 0

        def analyze_images(self, image_paths: list[Path], prompt: str) -> list[dict]:
            self.calls += 1
            return [{"i": 0, "d": "A visible lecture slide"}]

    client = FakeClient()
    warnings = describe_frames(observations, client, "English", cache_dir=tmp_path / "cache")  # type: ignore[arg-type]

    repeated = [
        VisualObservation("id", "Lecture", float(index), path) for index, path in enumerate(paths)
    ]
    second_warnings = describe_frames(repeated, client, "English", cache_dir=tmp_path / "cache")  # type: ignore[arg-type]

    assert not warnings and not second_warnings
    assert client.calls == 1
    assert observations[0].description == observations[1].description
    assert repeated[0].description == repeated[1].description
