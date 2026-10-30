"""Task 2: documents.jsonl -> chunks -> embeddings -> persistent Chroma (rebuilt from scratch)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.config import DOCUMENTS_PATH
from src.ingestion.chunking import chunk_document
from src.retrieval.vector import get_vector_store, reset_collection


def load_chunks() -> tuple[list[dict], int]:
    chunks, forced = [], 0
    for line in open(DOCUMENTS_PATH, encoding="utf-8"):
        doc_chunks, n_forced = chunk_document(json.loads(line))
        chunks += doc_chunks
        forced += n_forced
    return chunks, forced


def main():
    chunks, forced = load_chunks()
    by_type = Counter(c["metadata"]["doc_type"] for c in chunks)
    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} chunks, by doc_type: {dict(sorted(by_type.items()))}")
    print(f"chunk chars: min {min(sizes)}, max {max(sizes)}, avg {sum(sizes) // len(sizes)}")
    print(f"forced splits (over the size cap): {forced}")

    reset_collection()
    store = get_vector_store()
    store.add_documents(
        [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks],
        ids=[c["chunk_id"] for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks into Chroma.")


if __name__ == "__main__":
    main()
