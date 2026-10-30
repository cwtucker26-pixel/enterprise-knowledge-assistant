"""Run Task 1 ingestion: raw files -> data/processed/documents.jsonl and data/processed/acme.db."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCUMENTS_PATH, PROCESSED_DIR, RAW_DIR
from src.ingestion.loaders import load_faq, load_markdown, load_pdf
from src.ingestion.metadata import load_metadata_table
from src.ingestion.structured import build_sqlite


def main():
    meta_table = load_metadata_table()

    raw_docs = [load_pdf(p) for p in sorted((RAW_DIR / "pdfs").glob("*.pdf"))]
    raw_docs += [load_markdown(p) for p in sorted((RAW_DIR / "markdown").glob("*.md"))]
    raw_docs += load_faq(RAW_DIR / "faq" / "FAQ.md")

    documents = []
    for doc in raw_docs:
        metadata = dict(meta_table[doc.pop("source_file")])  # KeyError = file missing from the map
        extra = {k: doc.pop(k) for k in ("question", "faq_section") if k in doc}
        if extra:  # FAQ pairs: doc_metadata.csv calls FAQ.md "general"; we want "faq"
            metadata["doc_type"] = "faq"
            metadata.update(extra)
        documents.append({"doc_id": doc["doc_id"], "text": doc["text"], "metadata": metadata})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(documents)} documents -> {DOCUMENTS_PATH}")

    counts = build_sqlite()
    print("Wrote SQLite tables:", counts)


if __name__ == "__main__":
    main()
