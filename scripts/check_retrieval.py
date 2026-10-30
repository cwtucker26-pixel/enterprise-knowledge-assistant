"""Task 2 verification: run eval/test_questions.md through vector search and flag expected sources.

Prints the top-5 chunks per question and whether every expected source document appears.
Output is also written to eval/results/retrieval_baseline_vector.txt (baseline for Task 4).
Exit code is non-zero only if the index integrity checks fail.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CHUNK_MAX_CHARS, TOP_K
from src.retrieval.vector import get_vector_store, search
from scripts.build_index import load_chunks

QUESTIONS_PATH = ROOT / "eval" / "test_questions.md"
RESULTS_PATH = ROOT / "eval" / "results" / "retrieval_baseline_vector.txt"


class Tee:
    """Print to the terminal and collect the same lines for the results file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


def parse_questions() -> list[dict]:
    text = QUESTIONS_PATH.read_text(encoding="utf-8")
    questions = []
    for block in re.split(r"\n(?=## Q\d)", text)[1:]:
        qid = re.match(r"## (Q\d+)", block).group(1)
        question = re.search(r"\*\*Question:\*\*\s*(.+)", block).group(1).strip()
        sources = re.search(r"\*\*Sources:\*\*\s*(.+)", block).group(1)
        questions.append({"id": qid, "question": question,
                          "sources": [s.strip() for s in sources.split(",")]})
    return questions


def split_sources(sources: list[str]) -> tuple[list[str], list[str]]:
    """(expected doc ids/patterns, excluded because they are not in the vector index)."""
    expected, excluded = [], []
    for s in sources:
        (excluded if s.endswith(".csv") else expected).append(s)
    return expected, excluded


def source_found(source: str, doc_ids: list[str]) -> bool:
    if source == "FAQ.md":  # FAQ was split into FAQ-01..FAQ-11
        return any(d.startswith("FAQ-") for d in doc_ids)
    return source in doc_ids


def main() -> int:
    out = Tee()
    store = get_vector_store()
    out("Vector-only retrieval baseline (top %d per question; dist = cosine distance, lower is closer)" % TOP_K)

    coverage = []
    for q in parse_questions():
        results = search(store, q["question"], k=TOP_K)
        doc_ids = [doc.metadata["doc_id"] for doc, _ in results]
        expected, excluded = split_sources(q["sources"])
        found = [s for s in expected if source_found(s, doc_ids)]
        missing = [s for s in expected if s not in found]

        out(f"\n{q['id']}: {q['question']}")
        for doc, dist in results:
            first = doc.page_content.replace("\n", " ")[:80]
            out(f"  {doc.metadata['doc_id']:<24} {doc.metadata['chunk_id']:<26} {dist:.3f}  {first}")
        flag = "ALL FOUND" if not missing else "MISSING"
        out(f"  -> {flag}: {len(found)}/{len(expected)} expected sources in top {TOP_K}")
        if missing:
            out(f"     missing: {', '.join(missing)}")
        if excluded:
            out(f"     not in vector index (excluded): {', '.join(excluded)}")
        coverage.append((q["id"], len(found), len(expected)))

    out("\nSummary")
    for qid, f, e in coverage:
        out(f"  {qid}: {f}/{e}{'' if f == e else '  <-- incomplete'}")
    out(f"  questions fully covered: {sum(f == e for _, f, e in coverage)}/{len(coverage)}")
    out(f"  sources found overall:   {sum(f for _, f, _ in coverage)}/{sum(e for _, _, e in coverage)}")

    # Integrity: this runs in a fresh process, so the count also proves persistence.
    chunks, _ = load_chunks()
    stored = store.get()
    problems = []
    if len(stored["ids"]) != len(chunks):
        problems.append(f"collection has {len(stored['ids'])} chunks, expected {len(chunks)}")
    if len(set(stored["ids"])) != len(stored["ids"]):
        problems.append("duplicate chunk ids")
    if any(len(t) > CHUNK_MAX_CHARS + 200 for t in stored["documents"]):  # +200: header
        problems.append("a chunk exceeds the size cap")
    if {c["metadata"]["doc_id"] for c in chunks} != {m["doc_id"] for m in stored["metadatas"]}:
        problems.append("some documents have no chunks in the index")
    sop_only = search(store, "who do I escalate to", k=TOP_K, where={"doc_type": "sop"})
    if any(d.metadata["doc_type"] != "sop" for d, _ in sop_only):
        problems.append("doc_type filter returned non-SOP chunks")

    out("\nIntegrity: " + ("PASS" if not problems else "FAIL - " + "; ".join(problems)))
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"\n(written to {RESULTS_PATH.relative_to(ROOT)})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
