"""Retrieval check: run eval/test_questions.md through four configurations and compare source coverage.

  vector@5          saved baseline (eval/results/retrieval_baseline_vector.txt, not rewritten)
  vector@8          plain vector search, top 8
  vector+rerank@8   20 vector candidates, cross-encoder, top 8 (max 2 per doc)
  hybrid@8          vector + graph candidates, cross-encoder, top 8 (max 2 per doc)

Full hybrid detail is written to eval/results/retrieval_hybrid.txt.
Exit code is non-zero only if the index/hybrid integrity checks fail.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_index import load_chunks
from src.config import CHUNK_MAX_CHARS, MAX_CHUNKS_PER_DOC, RERANK_TOP_K
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector import search

QUESTIONS_PATH = ROOT / "eval" / "test_questions.md"
BASELINE_PATH = ROOT / "eval" / "results" / "retrieval_baseline_vector.txt"
RESULTS_PATH = ROOT / "eval" / "results" / "retrieval_hybrid.txt"


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
        questions.append({"id": qid, "question": question, "sources": [s.strip() for s in sources.split(",")]})
    return questions


def split_sources(sources: list[str]) -> tuple[list[str], list[str]]:
    """(expected doc ids, excluded because they are .csv files that are not in the vector index)."""
    expected = [s for s in sources if not s.endswith(".csv")]
    return expected, [s for s in sources if s.endswith(".csv")]


def source_found(source: str, doc_ids: list[str]) -> bool:
    if source == "FAQ.md":  # FAQ was split into FAQ-01..FAQ-11
        return any(d.startswith("FAQ-") for d in doc_ids)
    return source in doc_ids


def coverage(expected: list[str], doc_ids: list[str]) -> tuple[list[str], list[str]]:
    found = [s for s in expected if source_found(s, doc_ids)]
    return found, [s for s in expected if s not in found]


def saved_baseline() -> dict[str, tuple[int, int]]:
    """Per-question 'found/expected' from the saved vector@5 baseline file."""
    text = BASELINE_PATH.read_text(encoding="utf-8")
    return {m.group(1): (int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"^\s+(Q\d+): (\d+)/(\d+)", text, re.M)}


def main() -> int:
    out = Tee()
    retriever = HybridRetriever()
    baseline = saved_baseline()
    out(f"Hybrid retrieval check (top {RERANK_TOP_K}; score = cross-encoder logit, higher is better)")

    rows = []
    for q in parse_questions():
        expected, excluded = split_sources(q["sources"])
        vec_ids = [d.metadata["doc_id"] for d, _ in search(retriever.store, q["question"], k=RERANK_TOP_K)]
        vr = retriever.retrieve(q["question"], use_graph=False)
        hy = retriever.retrieve(q["question"])
        vr_ids = [d.metadata["doc_id"] for d in vr["chunks"]]
        hy_ids = [d.metadata["doc_id"] for d in hy["chunks"]]
        found_v, _ = coverage(expected, vec_ids)
        found_vr, _ = coverage(expected, vr_ids)
        found_h, missing_h = coverage(expected, hy_ids)

        out(f"\n{q['id']}: {q['question']}")
        out("  linked entities: " + (", ".join(e["name"] for e in hy["linked_entities"]) or "(none)"))
        st = hy["stats"]
        out(f"  candidate pool: {st['pool_size']} unique chunks ({st['vector_chunks']} vector + "
            f"{st['graph_chunks']} from {len(st['graph_docs'])} graph docs, overlap merged)")
        out("  graph docs: " + (", ".join(st["graph_docs"]) or "(none)"))
        out("  hybrid top 8:")
        for d in hy["chunks"]:
            first = d.page_content.replace("\n", " ")[:80]
            out(f"    {d.metadata['doc_id']:<24} {d.metadata['chunk_id']:<26} {d.metadata['rerank_score']:>6.2f}  "
                f"{d.metadata['origin']:<6} {first}")
        out(f"  graph facts ({len(hy['graph_facts'])}):")
        for f in hy["graph_facts"]:
            out(f"    - {f}")
        out(f"  vector@8 docs:        {', '.join(vec_ids)}")
        out(f"  vector+rerank@8 docs: {', '.join(vr_ids)}")
        out(f"  -> hybrid {len(found_h)}/{len(expected)} expected sources" + (f"; missing: {', '.join(missing_h)}" if missing_h else ""))
        if excluded:
            out(f"     not in vector index (excluded): {', '.join(excluded)}")
        rows.append({"id": q["id"], "n": len(expected), "v5": baseline.get(q["id"], (None, None))[0],
                     "v8": len(found_v), "vr8": len(found_vr), "h8": len(found_h),
                     "pool": st["pool_size"], "gdocs": len(st["graph_docs"])})

    out("\nSummary: expected sources found (top results)")
    out(f"  {'':3} {'expected':>8} {'vector@5':>9} {'vector@8':>9} {'vec+rerank@8':>13} {'hybrid@8':>9} {'pool':>6} {'graph docs':>11}")
    for r in rows:
        v5 = "-" if r["v5"] is None else str(r["v5"])
        out(f"  {r['id']:<3} {r['n']:>8} {v5:>9} {r['v8']:>9} {r['vr8']:>13} {r['h8']:>9} {r['pool']:>6} {r['gdocs']:>11}")
    tot = lambda k: sum(r[k] for r in rows if r[k] is not None)
    out(f"  {'all':<3} {tot('n'):>8} {tot('v5'):>9} {tot('v8'):>9} {tot('vr8'):>13} {tot('h8'):>9}")
    out(f"  questions fully covered: vector@5 {sum(r['v5'] == r['n'] for r in rows)}, vector@8 "
        f"{sum(r['v8'] == r['n'] for r in rows)}, vec+rerank@8 {sum(r['vr8'] == r['n'] for r in rows)}, "
        f"hybrid@8 {sum(r['h8'] == r['n'] for r in rows)} (of {len(rows)})")

    # Integrity: index matches a fresh chunking; hybrid output respects its limits.
    chunks, _ = load_chunks()
    stored = retriever.store.get()
    problems = []
    if len(stored["ids"]) != len(chunks):
        problems.append(f"collection has {len(stored['ids'])} chunks, expected {len(chunks)}")
    if len(set(stored["ids"])) != len(stored["ids"]):
        problems.append("duplicate chunk ids")
    if any(len(t) > CHUNK_MAX_CHARS + 200 for t in stored["documents"]):  # +200: header
        problems.append("a chunk exceeds the size cap")
    probe = retriever.retrieve(parse_questions()[1]["question"])["chunks"]
    ids = [d.metadata["chunk_id"] for d in probe]
    per_doc = [d.metadata["doc_id"] for d in probe]
    if len(ids) > RERANK_TOP_K or len(set(ids)) != len(ids):
        problems.append("hybrid returned too many or duplicate chunks")
    if any(per_doc.count(d) > MAX_CHUNKS_PER_DOC for d in set(per_doc)):
        problems.append("hybrid exceeded the per-document chunk limit")
    out("\nIntegrity: " + ("PASS" if not problems else "FAIL - " + "; ".join(problems)))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"\n(written to {RESULTS_PATH.relative_to(ROOT)})")
    retriever.close()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
