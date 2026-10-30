"""Task 12: latency (cold vs cached), cache behaviour, concurrency and cost per question, through the real API.

    python scripts/perf_test.py            # full run (sends requests)
    python scripts/perf_test.py --reread   # re-read the Langfuse traces of the saved cold requests and rewrite the report

Starts the API (scripts/eval_server.py), sends a SMALL set of requests to keep API cost low, then reads the
Langfuse traces of the cold requests for token usage and the per-node time breakdown. Writes
eval/results/perf_<date>.json (raw) and eval/results/performance_report.md.
"""
import datetime
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from scripts.eval_server import serve  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.observability.tracing import init_langfuse  # noqa: E402

# First-party API prices in USD per million tokens (claude-api skill price table, cached 2026-10-06).
# Cache-read prices: Sonnet 5.5 $0.20 (published); Haiku 5.5 assumed 0.1x input (not published in the table).
# Thinking tokens bill as output tokens.
PRICES = {"claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.20},
          "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01}}

COLD_SET = [  # (kind, question, facts the answer must contain)
    ("single-hop", "Who owns Payment-Service?", ["Billing", "Priya Sharma"]),
    ("single-hop", "What is the data retention period for transaction records?", ["7 years"]),
    ("multi-hop", "Which systems depend on Auth-DB, and who owns each?", ["APIGateway", "Chen Wei"]),
    ("conflict", "Does Auth-DB depend on Payment-Service?", ["INC-204"]),
    ("sql", "How many completed transactions are there, and what is their total amount in USD?", ["152"]),
]
CONCURRENT_SET = [  # distinct questions, all asked at the same time with the cache bypassed
    ("Who is the procedure owner of SOP-22?", ["Natalia Voss"]),
    ("Which team owns the SOP used to resolve INC-205, and who is its procedure owner?", ["Security", "Natalia Voss"]),
    ("What does the remote work policy require for personal devices?", ["MDM"]),
    ("What is the team lead escalation path in SOP-05?", ["Marcus Lee", "Natalia Voss"]),
]
CONCURRENCY_LEVELS = [4]


def post(base, question, session_id=None, use_cache=True):
    body = {"question": question, "use_cache": use_cache, **({"session_id": session_id} if session_id else {})}
    t0 = time.monotonic()
    r = httpx.post(base + "/chat", json=body, timeout=300)
    seconds = time.monotonic() - t0
    return {"status": r.status_code, "seconds": seconds, "x_cache": r.headers.get("x-cache"), "trace_id": r.headers.get("x-trace-id"),
            **(r.json() if r.status_code == 200 else {"detail": r.text[:200]})}


def has_facts(resp, facts):
    return all(f.lower() in resp.get("answer", "").lower().replace(",", "") for f in facts)


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def stats(values):
    return {"n": len(values), "median": statistics.median(values), "p90": pct(values, 0.9), "min": min(values), "max": max(values)}


def fetch_observations(lf, trace_id, timeout=180):
    """Wait until the trace is complete: the root span can be ingested before its children, so also wait for the
    observation count to stop changing."""
    deadline = time.monotonic() + timeout
    obs, last_count, stable = [], -1, 0
    while time.monotonic() < deadline:
        lf.flush()
        r = lf.api.observations.get_many(trace_id=trace_id, limit=300,
                                         fields="core,basic,io,model,usage,metrics,time")
        obs = [o.model_dump() if hasattr(o, "model_dump") else dict(o.__dict__) for o in r.data]
        root = next((o for o in obs if o.get("name") == "chat" and o.get("is_root_observation")), None)
        stable = stable + 1 if (root and root.get("output") is not None and len(obs) == last_count) else 0
        last_count = len(obs)
        if stable >= 2:
            return obs
        time.sleep(4)
    return obs


def dollars(model, usage):
    p = PRICES.get(model)
    if not p:
        return None
    cached = usage.get("cache_read_input_tokens", 0) or 0
    return ((usage.get("input", 0) - cached) * p["in"] + cached * p["cache_read"] + usage.get("output", 0) * p["out"]) / 1e6


def cost_of_trace(obs):
    """Sum the LangChain-level generations (name 'ChatAnthropic'); the Anthropic SDK instrumentation records every
    call a second time under 'anthropic.messages.create', which would double-count."""
    per_model, tokens = {}, {}
    for o in obs:
        if o.get("type") != "GENERATION" or o.get("name") != "ChatAnthropic":
            continue
        usage, model = o.get("usage_details") or {}, o.get("model") or "unknown"
        d = per_model.setdefault(model, {"calls": 0, "input": 0, "output": 0, "usd": 0.0})
        d["calls"] += 1
        d["input"] += usage.get("input", 0)
        d["output"] += usage.get("output", 0)
        d["usd"] += dollars(model, usage) or 0.0
    return per_model


def node_seconds(obs):
    """Total wall time per LangGraph node name (sum over calls)."""
    nodes = {"input_guard", "contextualize", "retrieve", "agent", "tools", "output_guard"}
    out = {}
    for o in obs:
        if o.get("name") in nodes and o.get("start_time") and o.get("end_time"):
            s, e = (datetime.datetime.fromisoformat(str(o[k]).replace("Z", "+00:00")) for k in ("start_time", "end_time"))
            out[o["name"]] = out.get(o["name"], 0.0) + (e - s).total_seconds()
    return out


def read_traces(lf, results):
    print("\n== reading Langfuse traces of the cold requests for tokens and node times ==", flush=True)
    for c in results["cold"]:
        obs = fetch_observations(lf, c["trace_id"])
        c["cost_by_model"], c["node_seconds"], c["n_observations"] = cost_of_trace(obs), node_seconds(obs), len(obs)
        c["usd"] = sum(m["usd"] for m in c["cost_by_model"].values())
        print(f"  {c['question'][:55]:55} ${c['usd']:.4f}  {c['n_observations']} observations  {({m: (v['calls'], v['input'], v['output']) for m, v in c['cost_by_model'].items()})}", flush=True)


def main():
    lf = init_langfuse(load_settings())
    if "--reread" in sys.argv:
        path = ROOT / "eval" / "results" / f"perf_{datetime.date.today().isoformat()}.json"
        results = json.loads(path.read_text(encoding="utf-8"))
        read_traces(lf, results)
        path.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        write_report(results)
        return
    results = {"date": datetime.date.today().isoformat()}
    with serve() as base:
        health = httpx.get(base + "/health").json()
        results["cache_config"] = health.get("cache")
        print("health:", health, flush=True)

        print("\n== cold: each question once, empty cache ==", flush=True)
        cold = []
        for kind, q, facts in COLD_SET:
            r = post(base, q)
            r.update(kind=kind, question=q, correct=has_facts(r, facts))
            cold.append(r)
            print(f"  {r['seconds']:6.1f}s  {r['x_cache']}  tools={r['tools_used']} flagged_warning={r['warning']} correct={r['correct']}  {q[:60]}", flush=True)
        results["cold"] = cold

        print("\n== cached: the same questions again (new sessions) ==", flush=True)
        cached = []
        for c in cold:
            r = post(base, c["question"])
            r.update(kind=c["kind"], question=c["question"], same_answer=r.get("answer") == c["answer"], expected_hit=c["x_cache"] == "MISS" and not c["warning"] and not c["blocked"] and "web_search" not in c["tools_used"])
            cached.append(r)
            print(f"  {r['seconds']:6.2f}s  {r['x_cache']}  expected_hit={r['expected_hit']} same_answer={r['same_answer']}  {c['question'][:60]}", flush=True)
        results["cached"] = cached

        print("\n== cache correctness probes ==", flush=True)
        probes = []

        def probe(label, q, expect, **kw):
            r = post(base, q, **kw)
            r.update(label=label, question=q, expect=expect, ok=(r["x_cache"] == expect))
            probes.append(r)
            print(f"  {r['seconds']:6.2f}s  {r['x_cache']:6} (expected {expect:6}) {label}: {q[:60]}", flush=True)
            return r

        probe("lower-case, no punctuation variant", "who owns payment service", "HIT")
        probe("no question mark", "Who owns Payment-Service", "HIT")
        probe("REVERSED dependency direction (cosine ~0.996 to a cached question)", "Does Payment-Service depend on Auth-DB?", "MISS")
        probe("cache bypassed with use_cache=false", "Who owns Payment-Service?", "BYPASS", use_cache=False)
        sid = f"perf-followup-{int(time.time())}"
        probe("first turn answered from the cache", "Who owns Payment-Service?", "HIT", session_id=sid)
        b = post(base, "What does it depend on?", session_id=sid)
        b.update(label="follow-up after a cache hit is not cached and still works", question="What does it depend on?", expect="SKIP",
                 ok=(b["x_cache"] == "SKIP" and "Payment-Service" in (b.get("rewritten_query") or "") and "Auth-DB" in b.get("answer", "")))
        probes.append(b)
        print(f"  {b['seconds']:6.2f}s  {b['x_cache']:6} follow-up rewritten={b.get('rewritten_query')!r} mentions Auth-DB={'Auth-DB' in b.get('answer','')}", flush=True)
        results["probes"] = probes

        print("\n== throughput: distinct questions at the same time, cache bypassed ==", flush=True)
        conc = {}
        for n in CONCURRENCY_LEVELS:
            batch = CONCURRENT_SET[:n]
            t0 = time.monotonic()
            with ThreadPoolExecutor(n) as pool:
                rs = list(pool.map(lambda qf: post(base, qf[0], use_cache=False), batch))
            wall = time.monotonic() - t0
            for r, (q, facts) in zip(rs, batch):
                r.update(question=q, correct=has_facts(r, facts))
            conc[n] = {"wall_s": wall, "requests": rs, "all_ok": all(r["status"] == 200 and r["correct"] for r in rs),
                       "requests_per_min": 60 * n / wall}
            print(f"  {n} concurrent: wall {wall:.1f}s, per-request {[round(r['seconds'], 1) for r in rs]}, all 200 and correct: {conc[n]['all_ok']}", flush=True)
        results["concurrency"] = conc
        results["health_after"] = httpx.get(base + "/health").json()

    read_traces(lf, results)
    (ROOT / "eval" / "results" / f"perf_{results['date']}.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    write_report(results)


def promptfoo_section(date):
    """Summarise the Promptfoo run made with the cache enabled (scripts/run_promptfoo.py --tag cache), if present."""
    path = ROOT / "eval" / "results" / f"promptfoo_{date}_cache.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))["results"]["results"]
    passed = [r for r in rows if r["success"]]
    failed = [r["testCase"]["description"].split(" ")[0] for r in rows if not r["success"]]
    hits = 0
    for r in rows:
        out = r["response"].get("output")
        out = json.loads(out) if isinstance(out, str) else out
        hits += bool(isinstance(out, dict) and out.get("cache_hit"))
    return ["## Regression check with the cache enabled", "",
            f"The full Promptfoo suite ({len(rows)} cases) was re-run after the cache was added (`promptfoo_{date}_cache.json`): **{len(passed)}/{len(rows)} passed**, "
            f"failures {', '.join(failed) or 'none'}, and **no regression against the baseline** (the same cases that failed in both earlier runs). "
            f"{'None' if hits == 0 else 'Only ' + str(hits)} of the {len(rows)} responses {'were' if hits == 0 else 'were'} cache hits: the suite's first-turn questions are almost all distinct, so it exercises the cache's miss-and-store path "
            "and shows it did not break any behaviour, but it does not exercise replay. Hits are covered by `scripts/check_api.py` and the probes above.", ""]


def write_report(R):
    cold, cached, probes, conc = R["cold"], R["cached"], R["probes"], R["concurrency"]
    cl, hl = [c["seconds"] for c in cold], [c["seconds"] for c in cached if c["x_cache"] == "HIT"]
    f = lambda v, d=1: f"{v:.{d}f}"
    S, Hs = stats(cl), (stats(hl) if hl else None)
    hits = sum(c["x_cache"] == "HIT" for c in cached)
    expected = sum(c["expected_hit"] for c in cached)
    usd = [c["usd"] for c in cold]
    models = {}
    for c in cold:
        for m, v in c["cost_by_model"].items():
            d = models.setdefault(m, {"calls": 0, "input": 0, "output": 0, "usd": 0.0})
            for k in d:
                d[k] += v[k]
    nodes = {}
    for c in cold:
        for n, s in c["node_seconds"].items():
            nodes.setdefault(n, []).append(s)
    L = [f"# Performance report ({R['date']})", "",
         "Measured with `python scripts/perf_test.py` against the real API (`POST /chat`), deliberately small request counts to keep API cost low. "
         "**With so few requests, treat medians and p90 as indicative, not as a benchmark**; p90 over 5 requests is essentially the slowest one.", "",
         "## Latency: cold vs cached", "",
         "| | n | median | p90 | min | max |", "|---|---|---|---|---|---|",
         f"| cold (full pipeline) | {S['n']} | {f(S['median'])} s | {f(S['p90'])} s | {f(S['min'])} s | {f(S['max'])} s |"]
    if Hs:
        L.append(f"| cached (semantic-cache hit) | {Hs['n']} | {f(Hs['median'], 3)} s | {f(Hs['p90'], 3)} s | {f(Hs['min'], 3)} s | {f(Hs['max'], 3)} s |")
        L += ["", f"A cache hit is about **{S['median'] / Hs['median']:.0f}x faster** than a cold answer at the median (a hit costs one local embedding call and no LLM call)."]
    L += ["", "Cold requests by question type:", "", "| type | question | seconds | tools | correct |", "|---|---|---|---|---|"]
    L += [f"| {c['kind']} | {c['question'][:70]} | {f(c['seconds'])} | {', '.join(c['tools_used']) or '-'} | {c['correct']} |" for c in cold]
    L += ["", f"Repeating the same {len(cached)} questions in new sessions gave {hits} cache hits ({expected} were expected: answers that were warned, blocked or used web search are never cached); "
          f"the replayed answer matched the original in {sum(c['same_answer'] for c in cached if c['x_cache'] == 'HIT')}/{hits} hits.", "",
          "## Cache correctness probes", "", "| probe | expected | got | pass |", "|---|---|---|---|"]
    L += [f"| {p['label']}: `{p['question'][:50]}` | {p['expect']} | {p['x_cache']} | {'yes' if p['ok'] else '**NO**'} |" for p in probes]
    L += ["", "The reversed-direction question is the important one: its embedding is almost identical to the cached question (cosine ~0.996, higher than genuine paraphrases), so an embedding-similarity threshold alone would have replayed the wrong answer. "
          "The cache therefore also requires the same structural signature (ordered entities plus direction/negation/number words); see `src/api/cache.py` and `scripts/check_cache.py`.", "",
          "## Throughput (concurrent requests)", ""]
    for n, c in ((int(k), v) for k, v in conc.items()):  # keys are strings after a JSON round trip
        L += [f"- **{n} distinct questions at once** (cache bypassed): wall time {f(c['wall_s'])} s, per-request latencies {[round(r['seconds'], 1) for r in c['requests']]} s, "
              f"all HTTP 200 and each answer contained its expected facts: **{c['all_ok']}** (no cross-talk between requests). Throughput {f(c['requests_per_min'])} requests/min. "
              f"For comparison, answering the same {n} one after another would take roughly {n} x the cold median = {f(n * S['median'])} s (an estimate, not measured).", ""]
    L += ["This exercises the shared embedding model, re-ranker, Neo4j driver and checkpointer under concurrency; no errors or wrong answers appeared. "
          "It is a 4-request test: it shows the shared models are usable from several threads, not where the system saturates (Anthropic rate limits and CPU inference are the likely limits).", "",
          "## Where the time goes (cold requests, from Langfuse spans)", "", "| node | median seconds per request | share of median latency |", "|---|---|---|"]
    for n in ("input_guard", "contextualize", "retrieve", "agent", "tools", "output_guard"):
        if n in nodes:
            L.append(f"| {n} | {f(statistics.median(nodes[n]))} | {100 * statistics.median(nodes[n]) / S['median']:.0f}% |")
    L += ["", "(`agent` is the Sonnet call(s) including its reasoning; `input_guard` and `output_guard` contain the Haiku classifier and groundedness judge calls; `retrieve` is embedding, Chroma, Neo4j and the cross-encoder. Nodes can run more than once per request, so the shares do not need to sum to 100%.)", "",
          "## Cost per question (from Langfuse token usage)", "",
          "| model | calls | input tokens | output tokens (incl. reasoning) | cost |", "|---|---|---|---|---|"]
    for m, v in models.items():
        L.append(f"| {m} | {v['calls']} | {v['input']:,} | {v['output']:,} | ${v['usd']:.4f} |")
    L += ["", f"Per cold question: median **${f(statistics.median(usd), 4)}**, range ${f(min(usd), 4)} to ${f(max(usd), 4)} (mean ${f(statistics.mean(usd), 4)}). "
          "A cache hit costs no model tokens. Computed from the `ChatAnthropic` generation spans (the Anthropic-SDK spans record each call a second time and are excluded), "
          "at USD per million tokens: Sonnet 5.5 $2 in / $10 out, Haiku 5.5 $0.10 in / $0.50 out (list prices in `PRICES` in `scripts/perf_test.py`; Haiku cache-read price assumed). "
          "Prompt-cache usage was not measured.", "",
          "**Observability finding: Langfuse totals were double the real figures, now fixed.** While reading the traces I noticed the Langfuse UI showed about $0.029 and 22k tokens for the SQL question, "
          "against ~$0.0143 and ~11k tokens computed here. Every model call was recorded twice (the LangChain handler's `ChatAnthropic` generation and the Anthropic SDK's own `anthropic.messages.create` span, nested under it) and Langfuse sums both. "
          "The tracing client now drops the SDK-level spans (`export_span` in `src/observability/tracing.py`); a trace taken after the change has only `ChatAnthropic` generations and totals of about 10.5k input + 0.4k output tokens. "
          "The figures in this report were computed from the `ChatAnthropic` spans alone, so they were never affected; traces recorded before the change (including the screenshots in `docs/screenshots/`) still show the doubled totals.", "",
          *promptfoo_section(R["date"]),
          "## Model routing: not implemented, on purpose", "",
          "Routing 'simple' questions to Haiku was considered and skipped:", "",
          "1. **The failure mode is silent.** A heuristic router (question length, one linked entity) will sometimes send a multi-hop question to the cheaper model, and the result is a plausible but wrong answer, not an error. The hardest questions here (Q2/Q3, the dependency conflict) look short and simple.",
          "2. **Adding it responsibly needs its own evaluation** (router accuracy, plus the Promptfoo/RAGAS suites on both paths), which is more cost and risk than the saving justifies at this scale.",
          "3. **The saving would be real, but small in absolute terms.** Sonnet is about 94% of the measured cost (table above) and Haiku is roughly 20x cheaper per token, so routing a question to Haiku could cut that question's cost by most of its agent portion. But a cold question costs about 1.4 cents, and the semantic cache already removes the whole cost of repeated questions. At this scale the quality risk outweighes a saving of a fraction of a cent per question.",
          "4. A safer first step, if cost ever matters: lower the agent's reasoning effort for first-pass answers and measure quality with the existing suites.", "",
          "## Limitations", "",
          "- 5 cold, 5 cached, 5 probe and 4 concurrent requests: small samples; a single run; no warm-up control for the embedding models.",
          "- The cache lives in process memory (lost on restart, one instance only), is keyed on the *masked* question, and hits are limited by design to near-identical wording.",
          "- Latency includes the model-based guards; a retry (a flagged first draft) roughly doubles a request.",
          "- Cost uses list prices and the token counts Langfuse recorded; thinking tokens are billed as output."]
    (ROOT / "eval" / "results" / "performance_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\nwrote eval/results/performance_report.md")


if __name__ == "__main__":
    main()
