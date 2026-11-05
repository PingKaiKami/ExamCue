from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .util import atomic_write_json

SUPPORTED_REFERENCE_SUFFIXES = {".pdf", ".pptx", ".txt", ".md"}
REFERENCE_INDEX_VERSION = 1
DEFAULT_RETRIEVAL_CHARS = 1200


@dataclass(slots=True)
class ReferenceCorpus:
    documents: list[dict]
    context: str
    hotwords: str
    chunks: list[dict]


def load_reference_corpus(
    paths: list[str],
    max_context_chars: int = 120000,
    cache_path: Path | None = None,
) -> ReferenceCorpus:
    fingerprint = _reference_fingerprint(paths)
    if cache_path is not None:
        cached = _load_cached_corpus(cache_path, fingerprint)
        if cached is not None:
            return cached

    documents: list[dict] = []
    parts: list[str] = []
    chunks: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        text = extract_reference_text(path)
        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        documents.append({"path": str(path), "name": path.name, "characters": len(cleaned)})
        if cleaned:
            parts.append(f"--- {path.name} ---\n{cleaned}")
            chunks.extend(_split_reference_chunks(path.name, cleaned))
    context = "\n\n".join(parts)[:max_context_chars]
    corpus = ReferenceCorpus(
        documents,
        context,
        _extract_hotwords("\n".join(parts)),
        chunks,
    )
    if cache_path is not None:
        atomic_write_json(
            cache_path,
            {
                "version": REFERENCE_INDEX_VERSION,
                "fingerprint": fingerprint,
                "documents": corpus.documents,
                "context": corpus.context,
                "hotwords": corpus.hotwords,
                "chunks": corpus.chunks,
            },
        )
    return corpus


def retrieve_reference_context(
    chunks: list[dict],
    query: str,
    limit: int = DEFAULT_RETRIEVAL_CHARS,
    max_chunks: int = 3,
) -> str:
    """Select a small, language-independent set of material excerpts.

    The index is lexical on purpose: it stays local, adds no embedding-model
    dependency, and works with mixed-script lecture text. Unrelated material is
    never inserted merely to fill the prompt budget.
    """
    query_terms = set(_search_terms(query))
    if not chunks or not query_terms or limit <= 0:
        return ""

    document_frequency: Counter[str] = Counter()
    chunk_terms: list[set[str]] = []
    for chunk in chunks:
        stored_terms = chunk.get("terms")
        terms = (
            {str(value) for value in stored_terms}
            if isinstance(stored_terms, list)
            else set(_search_terms(str(chunk.get("text") or "")))
        )
        chunk_terms.append(terms)
        document_frequency.update(terms)

    total = len(chunks)
    scored: list[tuple[float, int, dict]] = []
    for index, (chunk, terms) in enumerate(zip(chunks, chunk_terms, strict=True)):
        overlap = query_terms & terms
        if not overlap:
            continue
        score = sum(math.log((total + 1) / (document_frequency[term] + 1)) + 1 for term in overlap)
        score /= math.sqrt(max(len(terms), 1))
        scored.append((score, -index, chunk))

    selected: list[str] = []
    used = 0
    for _score, _order, chunk in sorted(scored, reverse=True):
        name = str(chunk.get("source") or "reference")
        text = str(chunk.get("text") or "").strip()
        excerpt = f"[{name}]\n{text}"
        remaining = limit - used
        if remaining <= 0 or len(selected) >= max_chunks:
            break
        if len(excerpt) > remaining:
            if not selected and remaining >= 160:
                excerpt = excerpt[:remaining].rstrip()
            else:
                continue
        selected.append(excerpt)
        used += len(excerpt) + 2
    return "\n\n".join(selected)


def _split_reference_chunks(source: str, text: str, max_chars: int = 900) -> list[dict]:
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    chunks: list[dict] = []
    current: list[str] = []
    current_chars = 0
    for paragraph in paragraphs:
        pieces = [
            paragraph[index : index + max_chars] for index in range(0, len(paragraph), max_chars)
        ]
        for piece in pieces:
            if current and current_chars + len(piece) + 2 > max_chars:
                text = "\n\n".join(current)
                chunks.append(
                    {"source": source, "text": text, "terms": sorted(set(_search_terms(text)))}
                )
                current = []
                current_chars = 0
            current.append(piece)
            current_chars += len(piece) + 2
    if current:
        text = "\n\n".join(current)
        chunks.append({"source": source, "text": text, "terms": sorted(set(_search_terms(text)))})
    return chunks


def _search_terms(text: str) -> list[str]:
    folded = text.casefold()
    words = re.findall(r"[^\W_]+(?:[._+-][^\W_]+)*", folded, flags=re.UNICODE)
    terms: list[str] = []
    for word in words:
        if len(word) >= 3:
            terms.append(word)
        if any(ord(character) > 127 for character in word):
            terms.extend(word[index : index + 2] for index in range(len(word) - 1))
    return terms


def _reference_fingerprint(paths: list[str]) -> str:
    state: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        stat = path.stat()
        state.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    value = f"{REFERENCE_INDEX_VERSION}|" + "|".join(state)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_cached_corpus(path: Path, fingerprint: str) -> ReferenceCorpus | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != REFERENCE_INDEX_VERSION
            or payload.get("fingerprint") != fingerprint
        ):
            return None
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            return None
        return ReferenceCorpus(
            list(payload.get("documents") or []),
            str(payload.get("context") or ""),
            str(payload.get("hotwords") or ""),
            [value for value in chunks if isinstance(value, dict)],
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def extract_reference_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Reference document not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_REFERENCE_SUFFIXES:
        raise ValueError(f"Unsupported reference format: {suffix}")
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdf") from exc
        return "\n\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError("PowerPoint support requires python-pptx") from exc
    slides: list[str] = []
    for slide in Presentation(path).slides:
        values = [shape.text for shape in slide.shapes if hasattr(shape, "text")]
        slides.append("\n".join(value for value in values if value.strip()))
    return "\n\n".join(slides)


def _extract_hotwords(text: str, limit: int = 80) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{3,}|[\u3400-\u9fff]{2,8}", text)
    ignored = {"this", "that", "with", "from", "have", "will", "可以", "這個", "我們"}
    counts = Counter(token for token in tokens if token.casefold() not in ignored)
    ranked = sorted(counts, key=lambda value: (-counts[value], -len(value), value.casefold()))
    return ", ".join(ranked[:limit])
