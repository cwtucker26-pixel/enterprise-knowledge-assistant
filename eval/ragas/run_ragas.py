"""Task 10: RAGAS evaluation of the real system.

1. Answers come from the real API (POST /chat, server started by scripts/eval_server.py).
2. Contexts are rebuilt with the same retriever code (retrieval is deterministic) and checked against the chunk ids
   the API reported. Contexts = the retrieved chunk texts PLUS the graph facts (as one extra context item), because
   the answer relies on both; leaving the facts out would unfairly penalise faithfulness.
3. Metrics: faithfulness, answer relevancy (hybrid answers, with chunks+facts as context), and context precision /
   context recall for three retrieval variants: hybrid (chunks + facts), hybrid chunks only, and vector-only top 8.

    python eval/ragas/run_ragas.py [--judge claude-sonnet-5-5]
    python eval/ragas/run_ragas.py --complete [--dry-run]   # score only what is missing in the latest saved run

`--complete` exists because a run can be interrupted (the first run was, by an exhausted API balance): it re-scores the
entries that are None in the saved JSON and adds the `hybrid_facts_first` variant (the same contexts with the graph
facts item FIRST, as in the real prompt; context precision is rank-sensitive, and the first run put the facts last).
"""
import argparse
import datetime
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import httpx  # noqa: E402
from judge import make_judge  # noqa: E402  (applies the RAGAS compatibility shim first)
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,  # noqa: E402
                           ResponseRelevancy)
from ragas.run_config import RunConfig  # noqa: E402

from scripts.eval_server import serve  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

DATASET = HERE / "dataset.json"
TOP_K_VECTOR = 8


def ask(base_url: str, question: str) -> dict:
    r = httpx.post(base_url + "/chat", json={"question": question}, timeout=300)
    return {"status": r.status_code, **r.json()}


def facts_item(facts: list[str]) -> list[str]:
    return ["Graph facts from the knowledge graph:\n" + "\n".join(f"- {f}" for f in facts)] if facts else []


def run_metrics(name, samples, metrics, llm, emb) -> list[dict]:
    print(f"RAGAS: {name} ({len(samples)} samples, metrics: {[m.name for m in metrics]})", flush=True)
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics, llm=llm, embeddings=emb,
                      run_config=RunConfig(max_workers=8, timeout=600, max_retries=3), show_progress=False)
    df = result.to_pandas()
    cols = [m.name for m in metrics]
    return [{c: (None if (v := row[c]) is None or (isinstance(v, float) and math.isnan(v)) else float(v)) for c in cols}
            for _, row in df.iterrows()]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def facts_first(row: dict) -> list[str]:
    ctx = row["ctx_hybrid"]
    return ctx[-1:] + ctx[:-1] if row["n_facts"] else ctx


def complete(args):
    """Fill the None scores of the latest saved run, and add the facts-first variant."""
    path = ROOT / "eval" / "results" / f"ragas_{datetime.date.today().isoformat()}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    rows = saved["rows"]
    prec_key, rec_key = "llm_context_precision_with_reference", "context_recall"
    variants = {  # name -> (contexts for a row, needs the answer, metric keys)
        "hybrid": (lambda r: r["ctx_hybrid"], True, ["faithfulness", "answer_relevancy", prec_key, rec_key]),
        "hybrid_chunks_only": (lambda r: r["ctx_chunks"], False, [prec_key, rec_key]),
        "vector_only": (lambda r: r["ctx_vector"], False, [prec_key, rec_key]),
        "hybrid_facts_first": (facts_first, False, [prec_key, rec_key]),
    }
    todo = {}
    for name, (_, _, keys) in variants.items():
        for i, r in enumerate(rows):
            have = r["scores"].get(name, {})
            if any(have.get(k) is None for k in keys):
                todo.setdefault(name, []).append(i)
    for name, idx in todo.items():
        print(f"{name}: {len(idx)} question(s) to score: {[rows[i]['id'] for i in idx]}")
    if args.dry_run or not todo:
        print("dry run / nothing to do: no API calls made")
        return
    llm, emb = make_judge(args.judge)
    by_key = {"faithfulness": Faithfulness(), "answer_relevancy": ResponseRelevancy(),
              prec_key: LLMContextPrecisionWithReference(), rec_key: LLMContextRecall()}
    for name, idx in todo.items():
        ctx_of, needs_answer, keys = variants[name]
        samples = [SingleTurnSample(user_input=rows[i]["question"], response=rows[i]["answer"] if needs_answer else None,
                                    retrieved_contexts=ctx_of(rows[i]), reference=rows[i]["reference"]) for i in idx]
        scores = run_metrics(name, samples, [by_key[k] for k in keys], llm, emb)
        for i, sc in zip(idx, scores):
            rows[i]["scores"].setdefault(name, {}).update({k: v for k, v in sc.items() if v is not None})
            for k in keys:
                rows[i]["scores"][name].setdefault(k, None)
    path.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
    print("updated", path.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None, help="judge model (default: the reasoning model)")
    ap.add_argument("--complete", action="store_true", help="score only what is missing in the latest saved run")
    ap.add_argument("--dry-run", action="store_true", help="with --complete: list what would be scored, call nothing")
    args = ap.parse_args()
    if args.complete:
        return complete(args)
    data = json.loads(DATASET.read_text(encoding="utf-8"))["questions"]

    with serve() as base_url, ThreadPoolExecutor(3) as pool:
        answers = list(pool.map(lambda q: ask(base_url, q["question"]), data))

    retriever = HybridRetriever()
    rows = []
    for q, a in zip(data, answers):
        r = retriever.retrieve(q["question"])
        chunks = [d.page_content for d in r["chunks"]]
        vector = [d.page_content for d in retriever.store.similarity_search(q["question"], k=TOP_K_VECTOR)]
        api_ids = [s["chunk_id"] for s in a.get("sources", [])]
        rows.append({**q, "answer": a.get("answer", ""), "status": a["status"], "blocked": a.get("blocked"),
                     "warning": a.get("warning"), "tools_used": a.get("tools_used"),
                     "chunk_ids": [d.metadata["chunk_id"] for d in r["chunks"]], "api_chunk_ids": api_ids,
                     "contexts_match_api": api_ids == [d.metadata["chunk_id"] for d in r["chunks"]],
                     "ctx_hybrid": chunks + facts_item(r["graph_facts"]), "ctx_chunks": chunks, "ctx_vector": vector,
                     "n_facts": len(r["graph_facts"])})
    retriever.close()
    print(f"contexts rebuilt in-process match the API's sources for {sum(r['contexts_match_api'] for r in rows)}/{len(rows)} questions")

    llm, emb = make_judge(args.judge)
    judge_name = args.judge or "reasoning model"

    def samples(ctx_key, with_response):
        return [SingleTurnSample(user_input=r["question"], response=r["answer"] if with_response else None,
                                 retrieved_contexts=r[ctx_key], reference=r["reference"]) for r in rows]

    prec, rec = LLMContextPrecisionWithReference(), LLMContextRecall()
    full = run_metrics("hybrid, all four metrics", samples("ctx_hybrid", True),
                       [Faithfulness(), ResponseRelevancy(), prec, rec], llm, emb)
    chunks_only = run_metrics("hybrid chunks only", samples("ctx_chunks", False), [prec, rec], llm, emb)
    vector_only = run_metrics("vector-only top 8", samples("ctx_vector", False), [prec, rec], llm, emb)

    for r, f, c, v in zip(rows, full, chunks_only, vector_only):
        r["scores"] = {"hybrid": f, "hybrid_chunks_only": c, "vector_only": v}

    out = ROOT / "eval" / "results" / f"ragas_{datetime.date.today().isoformat()}.json"
    out.write_text(json.dumps({"date": datetime.date.today().isoformat(), "judge": judge_name, "rows": rows}, indent=1,
                              ensure_ascii=False), encoding="utf-8")

    cats = sorted({r["category"] for r in rows})
    names = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy",
             "llm_context_precision_with_reference": "context_precision", "context_recall": "context_recall"}
    lines = [f"Judge: {judge_name}. Questions: {len(rows)}.", "", "| metric | hybrid (chunks+facts) | hybrid chunks only | vector-only top 8 |", "|---|---|---|---|"]
    for key, label in names.items():
        cells = [mean([r["scores"][v].get(key) for r in rows]) for v in ("hybrid", "hybrid_chunks_only", "vector_only")]
        lines.append(f"| {label} | " + " | ".join("n/a" if x is None else f"{x:.3f}" for x in cells) + " |")
    print("\n".join(lines))
    for cat in cats:
        sel = [r for r in rows if r["category"] == cat]
        print(cat, {label: round(mean([r["scores"]["hybrid"].get(key) for r in sel]) or -1, 3) for key, label in names.items()})
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
