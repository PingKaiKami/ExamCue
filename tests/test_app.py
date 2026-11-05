from pathlib import Path

from examcue.app import (
    RESULT_COLUMN_MIN_WIDTHS,
    _filter_media_paths,
    _filter_reference_paths,
    _result_column_widths,
)


def test_result_column_widths_keep_fixed_columns_and_fill_available_space():
    widths = _result_column_widths(1070)

    assert widths == (125, 105, 72, 252, 70, 446)
    assert sum(widths) == 1070


def test_result_column_widths_never_shrink_below_minimums():
    assert _result_column_widths(700) == RESULT_COLUMN_MIN_WIDTHS


def test_reference_drop_accepts_multiple_supported_files(tmp_path: Path):
    pdf = tmp_path / "slides one.pdf"
    pptx = tmp_path / "lesson.pptx"
    unsupported = tmp_path / "notes.docx"
    for path in (pdf, pptx, unsupported):
        path.write_bytes(b"test")

    accepted, rejected = _filter_reference_paths([str(pdf), str(pptx), str(pdf), str(unsupported)])

    assert accepted == [str(pdf.resolve()), str(pptx.resolve())]
    assert rejected == ["notes.docx"]


def test_media_drop_accepts_multiple_supported_files(tmp_path: Path):
    video = tmp_path / "lesson one.mp4"
    audio = tmp_path / "lecture.mp3"
    unsupported = tmp_path / "slides.pdf"
    for path in (video, audio, unsupported):
        path.write_bytes(b"test")

    accepted, rejected = _filter_media_paths([str(video), str(audio), str(video), str(unsupported)])

    assert accepted == [str(video.resolve()), str(audio.resolve())]
    assert rejected == ["slides.pdf"]
