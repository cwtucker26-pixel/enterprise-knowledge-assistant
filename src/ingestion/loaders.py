"""Load raw files into plain documents: {"doc_id", "text", "metadata"} (metadata added later)."""
import re
from pathlib import Path

from pypdf import PdfReader

from src.ingestion.cleaning import clean_markdown_text, clean_pdf_text


def load_pdf(path: Path) -> dict:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return {"doc_id": path.stem, "text": clean_pdf_text(text), "source_file": path.name}


def load_markdown(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return {"doc_id": path.stem, "text": clean_markdown_text(text), "source_file": path.name}


def load_faq(path: Path) -> list[dict]:
    """One document per Q&A pair. Tracks the '## Section' heading above each pair."""
    docs, section = [], None
    for block in re.split(r"\n(?=## |\*\*Q:)", path.read_text(encoding="utf-8")):
        block = block.strip()
        if block.startswith("## "):
            section = block[3:].strip()
        elif block.startswith("**Q:"):
            question = re.match(r"\*\*Q:\s*(.*?)\*\*", block, re.S).group(1).strip()
            answer = block.split("\nA:", 1)[1].strip()
            n = len(docs) + 1
            docs.append({
                "doc_id": f"FAQ-{n:02d}",
                "text": clean_markdown_text(f"Q: {question}\nA: {answer}"),
                "source_file": path.name,
                "question": question,
                "faq_section": section,
            })
    return docs
