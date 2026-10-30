"""Section-aware chunking. Each chunk = a context header + one section (or a few small ones).

FAQ pairs stay whole. Markdown splits on '#'/'##' headings, PDFs on '1. Heading' lines.
Sections under CHUNK_MIN_CHARS merge into a neighbour; sections over CHUNK_MAX_CHARS are split
on line boundaries with a small overlap (the only place overlap is used).
"""
import re

from src.config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_CHARS

MD_HEADING = re.compile(r"^#{1,2}\s+(.*)")
PDF_HEADING = re.compile(r"^\d+\.\s+([A-Z].*)")
PREAMBLE = "Overview"  # title + header fields that appear before the first heading


def split_sections(text: str, is_pdf: bool) -> list[tuple[str, str]]:
    """Return [(section_title, section_text)]. The heading line stays inside section_text."""
    heading_re = PDF_HEADING if is_pdf else MD_HEADING
    sections, title, lines = [], PREAMBLE, []
    for line in text.split("\n"):
        m = heading_re.match(line)
        # A markdown '# Title' on the first line is part of the preamble, not a new section.
        if m and not (not is_pdf and line.startswith("# ") and not lines and title == PREAMBLE):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title, lines = m.group(1).strip(), [line]
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    return [(t, s) for t, s in sections if s]


def join_titles(a: str, b: str) -> str:
    return a if a == b else f"{a} + {b}"


def merge_small(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge a section under CHUNK_MIN_CHARS into the next one (the last one into the previous)."""
    merged = []
    for title, text in sections:
        if merged and len(merged[-1][1]) < CHUNK_MIN_CHARS:
            prev_title, prev_text = merged.pop()
            if len(prev_text) + len(text) + 2 <= CHUNK_MAX_CHARS:
                merged.append((join_titles(prev_title, title), prev_text + "\n\n" + text))
                continue
            merged.append((prev_title, prev_text))
        merged.append((title, text))
    if len(merged) > 1 and len(merged[-1][1]) < CHUNK_MIN_CHARS:
        title, text = merged.pop()
        prev_title, prev_text = merged[-1]
        if len(prev_text) + len(text) + 2 <= CHUNK_MAX_CHARS:
            merged[-1] = (join_titles(prev_title, title), prev_text + "\n\n" + text)
        else:
            merged.append((title, text))
    return merged


def force_split(text: str) -> list[str]:
    """Split an over-long section on line boundaries, carrying <= CHUNK_OVERLAP_CHARS of tail."""
    pieces, current = [], []
    for line in text.split("\n"):
        if current and len("\n".join(current + [line])) > CHUNK_MAX_CHARS:
            pieces.append("\n".join(current))
            tail, size = [], 0
            for prev in reversed(current):
                if size + len(prev) > CHUNK_OVERLAP_CHARS:
                    break
                tail.insert(0, prev)
                size += len(prev) + 1
            current = tail
        current.append(line)
    pieces.append("\n".join(current))
    return pieces


def make_header(doc: dict, section: str) -> str:
    meta = doc["metadata"]
    parts = [doc["doc_id"], meta["doc_type"]]
    if meta["system_refs"]:
        parts.append(", ".join(meta["system_refs"]))
    parts.append(section)
    return "[" + " | ".join(parts) + "]"


def flatten_metadata(meta: dict) -> dict:
    """Chroma metadata accepts only str/int/float/bool: lists become comma-joined strings."""
    return {k: ",".join(v) if isinstance(v, list) else v for k, v in meta.items() if v is not None}


def chunk_document(doc: dict) -> tuple[list[dict], int]:
    """Return ([{chunk_id, text, metadata}], number_of_forced_splits)."""
    meta = doc["metadata"]
    if meta["doc_type"] == "faq":
        sections = [(meta.get("faq_section") or "FAQ", doc["text"])]
    else:
        is_pdf = meta["source_file"].endswith(".pdf")
        sections = merge_small(split_sections(doc["text"], is_pdf))

    chunks, forced = [], 0
    for title, text in sections:
        pieces = force_split(text) if len(text) > CHUNK_MAX_CHARS else [text]
        forced += len(pieces) - 1
        for piece in pieces:
            n = len(chunks)
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{n}",
                "text": make_header(doc, title) + "\n" + piece,
                "metadata": {
                    **flatten_metadata(meta),
                    "doc_id": doc["doc_id"],
                    "chunk_id": f"{doc['doc_id']}#{n}",
                    "chunk_index": n,
                    "section": title,
                },
            })
    return chunks, forced
