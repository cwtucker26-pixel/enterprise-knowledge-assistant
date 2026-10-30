# Enterprise Knowledge Assistant

A production-style GenAI backend that answers employee questions over a fictional company's (ACME Corp) IT-operations documents.
Capstone project, Module 14 of an AI Engineering course.

It combines **hybrid retrieval** (vector search plus a Neo4j knowledge graph, re-ranked by a cross-encoder), a **LangGraph agent** with
tools (read-only SQL, web search, Python), **layered guardrails** (prompt-injection detection, PII masking, citation and groundedness
checks, leak filtering), a **FastAPI** backend with sessions and a semantic cache, **Langfuse** tracing, and automated evaluation
(**Promptfoo** regression suite, **RAGAS**, a safety scorecard).

Start with [docs/walkthrough.md](docs/walkthrough.md) for the story and the findings, [docs/architecture.md](docs/architecture.md) for the design.

## Example

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "Which systems depend on Auth-DB, and who owns each?"}'
```

returns the answer (APIGateway, DataWarehouse, Payment-Service, UserProfile-API with their owner teams and leads, cited to the manuals), the
parsed `citations`, the retrieved `sources` (chunk ids and document ids), the `tools_used`, every guard's result, the rewritten query for
follow-ups, `cache_hit`, and `latency_ms`.

## Setup from a fresh clone

**Prerequisites:** Python 3.12, Node.js 18+ (only for the Promptfoo suite), and four external accounts (all have free tiers):
an Anthropic API key, a Neo4j AuraDB Free instance, a Tavily API key, and a Langfuse Cloud project. The embedding model (about 130 MB) and
the re-ranker (about 90 MB) download from Hugging Face on first use.

```bash
git clone <repo-url> enterprise-knowledge-assistant && cd enterprise-knowledge-assistant
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # versions are pinned to the ones the project was evaluated with
npm install                                # Promptfoo (Node), only needed for the regression suite

cp .env.example .env                       # then fill in the eight values; .env is git-ignored, never commit it
python scripts/check_setup.py              # PASS/FAIL per service; never prints secrets
```

**Build the data stores** (the raw dataset is in `data/raw/`; derived data is git-ignored and rebuilt by these scripts):

```bash
python scripts/run_ingestion.py            # data/processed/documents.jsonl (32 documents) and acme.db (SQLite: users, products, transactions)
python scripts/build_index.py              # chunks (107), embeds, rebuilds the ChromaDB index in chroma_db/
python scripts/build_graph.py              # extracts the graph and reloads Neo4j (clears the instance first: use a dedicated one)
```

Each stage has a verification script that needs no LLM calls: `check_ingestion.py`, `check_retrieval.py` (vector and hybrid retrieval on the
test questions, writes `eval/results/retrieval_hybrid.txt`), `check_graph.py` (Cypher queries for the multi-hop questions), `check_cache.py`.

## Run the API

```bash
uvicorn src.api.main:app --port 8000
```

- Interactive docs (try requests in the browser): http://localhost:8000/docs
- Health: `curl localhost:8000/health` (Chroma, Neo4j, agent loaded, cache stats)
- Startup loads the embedding model, the re-ranker and the guard models once (10 to 20 seconds).

`POST /chat` takes `{"question": "...", "session_id": "optional", "use_cache": true}`. Reuse the returned `session_id` for follow-ups
("What does it depend on?" is rewritten using the conversation). Response headers: `X-Trace-Id` (the Langfuse trace) and `X-Cache`
(`HIT`, `MISS`, `SKIP` for follow-ups, `BYPASS` when disabled). Disable the semantic cache with `CACHE_ENABLED=false` or per request with
`"use_cache": false`. Ask from the terminal without the API: `python scripts/ask.py "Who owns Payment-Service?"`.

Typical latency: cold median 8.4 s (p90 15.3 s, a handful of requests), cache hit about 0.03 s.

## Evaluate

All evaluations call the real API over HTTP: each script starts uvicorn on port 8765, runs, and stops it (`scripts/eval_server.py`).
Results go to `eval/results/` with the run date. They call the Anthropic API, so they cost money: a cold question costs about $0.014 (see the
performance report), so a full 75-case Promptfoo run is on the order of $1 (an estimate from that figure, not a measurement).

```bash
python scripts/run_promptfoo.py            # regression suite: 75 cases, compared with eval/promptfoo/baseline.json
python scripts/run_safety.py               # 49 adversarial/benign prompts -> eval/safety/safety_scorecard.md
python eval/ragas/run_ragas.py             # RAGAS faithfulness / answer relevancy / context precision / recall (18 questions)
python scripts/perf_test.py                # latency cold vs cached, concurrency, cost per question
python scripts/check_agent.py              # tool-safety tests (no LLM) plus agent questions
python scripts/check_guardrails.py         # guardrail unit tests plus live injection / PII / false-positive cases
python scripts/check_api.py                # API, sessions, cache and Langfuse trace checks (fetches traces back from Langfuse)
```

**Regression test (one command):** `python scripts/run_promptfoo.py` exits non-zero if a case that passed in the baseline (the 72 cases that
passed in two full runs) fails now. `--update-baseline` accepts a new run, `--filter 'SH-|MH-'` runs a subset, `--tag name` keeps the results
file. Answers vary between runs, so the cases assert facts, fields and behaviour (blocked, tools used, citations, no raw PII), not wording.

## Results at a glance

| | Result | Details |
|---|---|---|
| Promptfoo (75 cases) | 72/75 in two full runs and in a third with the cache on; the same 3 fail each time | [failure analysis](eval/results/promptfoo_failure_analysis.md) |
| RAGAS (18 questions) | faithfulness 0.92, answer relevancy 0.92, context precision 0.72 to 0.74, context recall 0.91 (0.69 vector-only) | [report](eval/results/ragas_report.md) |
| Safety (49 prompts) | 39/39 adversarial handled, 1/10 benign blocked, 0 leaks | [scorecard](eval/safety/safety_scorecard.md) |
| Performance | cold median 8.4 s, cache hit 0.026 s, 4 concurrent requests in 12.6 s, about $0.014 per cold question | [report](eval/results/performance_report.md) |

These are small, mostly self-authored evaluations with caveats written next to each number; read the linked files before quoting them.

## How it was built (stepwise)

| Step | What | Where |
|---|---|---|
| 0 | Project setup: venv, dependencies, `.env`, service connectivity check | `scripts/check_setup.py` |
| 1 | Ingestion: load PDFs, markdown and FAQ into one document format, clean lightly, join metadata, CSVs into SQLite | `src/ingestion/`, `scripts/run_ingestion.py` |
| 2 | Vector index: section-aware chunks with context headers, bge embeddings, ChromaDB | `src/ingestion/chunking.py`, `src/retrieval/vector.py` |
| 3 | Knowledge graph: deterministic extraction, Neo4j, source and authority on every edge | `src/ingestion/graph_extract.py`, `src/retrieval/graph.py` |
| 4 | Hybrid retrieval: entity linking, graph expansion, merge, cross-encoder re-rank, graph facts | `src/retrieval/hybrid.py`, `rerank.py` |
| 5 | LangGraph agent with SQL, web and Python tools and a max-steps guard | `src/agent/` |
| 6 | Guardrails: input validation, injection detection, PII masking, citation and groundedness checks, leak filter | `src/guardrails/` |
| 7 | FastAPI: `/chat`, sessions, follow-up rewriting, structured responses with citations | `src/api/` |
| 8 | Langfuse tracing: node spans, tokens, tags, metadata, masked PII, errors | `src/observability/tracing.py` |
| 9 to 11 | Promptfoo suite, RAGAS evaluation, safety scorecard | `eval/`, `scripts/run_*.py` |
| 12 | Semantic cache, latency / concurrency / cost measurements | `src/api/cache.py`, `scripts/perf_test.py` |
| 13 | Documentation, pinned dependencies | `docs/`, `README.md`, `requirements.txt` |

## Project layout

```
data/raw/            the provided dataset (pdfs, markdown, faq, structured CSVs, doc_metadata.csv)
data/processed/      generated: documents.jsonl, acme.db (git-ignored)
src/ingestion/       loaders, cleaning, chunking, metadata, graph extraction
src/retrieval/       vector.py, graph.py, hybrid.py, rerank.py
src/agent/           state.py, graph.py, tools.py, history.py, answer_chain.py
src/guardrails/      input.py, output.py, pii.py
src/api/             main.py, schemas.py, cache.py
src/observability/   tracing.py (Langfuse)
scripts/             build, check_*, run_* (evaluation), perf_test.py, eval_server.py
eval/                promptfoo/ (suite), ragas/, safety/ (dataset, scorecard), results/ (dated outputs), test_questions.md
docs/                architecture.md, walkthrough.md, agent_graph.md, screenshots/
```

## Limitations

- **`python_exec` is a subprocess with a timeout, an empty environment and a temp directory, not a sandbox.**
- **No authentication; sessions and the cache live in process memory** (lost on restart, never expired, single worker only). A `session_id` acts as a bearer token.
- **The model-based guards fail open** on API errors; injection detection is heuristic, with known false positives.
- **Retrieval gaps**: the PII-breach question never retrieves SOP-22, and there is no lexical (BM25) search.
- **Evaluation is small** (18 RAGAS questions with self-written references and a same-family judge; a self-written adversarial set; few performance requests).
- Model routing and an MCP tool server were not built; see [docs/walkthrough.md](docs/walkthrough.md) for the reasoning and the improvement list.
