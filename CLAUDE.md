# Enterprise Knowledge Assistant — Project Guide for Claude Code

Capstone project (Module 14, AI Engineering course). A production-style GenAI backend that answers
employee questions over a fictional company's documents using hybrid RAG (vector + graph),
a LangGraph agent with tools, multi-layered guardrails, FastAPI, Langfuse observability, and
automated evaluation (Promptfoo, RAGAS, safety scorecard).

The developer (Christopher) is learning. Before implementing anything new, explain the concept
in a few sentences and how it fits the overall system. Prefer simple, readable code over clever code.

## Working agreement
- Work one phase at a time. Start each phase in plan mode; wait for approval before writing code.
- Don't add tools, libraries, or features beyond what's listed here without asking.
- Every phase ends with a small verification script or command that proves it works.
- Keep modules small and single-purpose. Config is read only through `src/config.py`.
- Never commit `.env`. Never print secrets in logs.
- Update the checklist below when a task is complete.

## Locked stack decisions
| Concern | Choice |
|---|---|
| Python | 3.11 or 3.12, venv at `.venv/` |
| LLM | Anthropic API via `langchain-anthropic` (`claude-sonnet-5-5` for reasoning, `claude-haiku-5-5` for simple/cheap calls) |
| Embeddings | Local `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Vector DB | ChromaDB (persistent, local folder `chroma_db/`) |
| Graph DB | Neo4j AuraDB Free |
| Re-ranker | Local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Orchestration | LangChain (LCEL chains inside nodes) + LangGraph (workflow) |
| Tools | Tavily web search, SQL query tool (SQLite over the CSVs), Python execution (subprocess + timeout, never `exec()`) |
| API | FastAPI + Uvicorn |
| Observability | Langfuse Cloud |
| Testing/Eval | Promptfoo (Node), RAGAS, custom safety eval script |

## Fictional enterprise: Halcyon VFX
A mid-size visual effects studio. Dummy data lives in `data/raw/`. It should be internally
consistent and include:
- **PDFs (policies, manuals):** IT security policy, data handling/client confidentiality policy,
  render farm operations manual, remote work policy, vendor/outsourcing policy (~5-6 files)
- **Markdown (SOPs, incident reports, KB articles):** render farm outage escalation, license
  server recovery, asset publish SOP, shot delivery SOP, 3-4 incident reports (~10-12 files)
- **FAQ:** ~20 Q&A pairs (single markdown or JSON file)
- **CSVs:** employees, systems, shots, tickets (small: 20-50 rows each)
- **Graph seed:** entities (people, teams, systems, software, shows) and relationships
  (`DEPENDS_ON`, `OWNS`, `RELATED_TO`, `ASSIGNED_TO`, `MEMBER_OF`)

Deliberately include:
- Facts split across documents so multi-hop questions need the graph
  (e.g. "Who do I contact if the system Nuke depends on goes down?")
- Some fake PII (emails, phone numbers) in a few docs to exercise PII masking
- A few near-duplicate/conflicting passages to exercise re-ranking

## Project structure
```
data/raw/{pdfs,markdown,faq,structured,graph}/  data/processed/
src/config.py
src/ingestion/      loaders, cleaning, chunking, metadata
src/retrieval/      vector.py, graph.py, hybrid.py, rerank.py
src/agent/          state.py, graph.py, tools.py
src/guardrails/     input.py, output.py
src/api/main.py
src/observability/
eval/{promptfoo,ragas,safety}/
scripts/            generate_data.py, build_index.py, build_graph.py, ask.py
docs/               architecture.md, screenshots/
```

## Requirements checklist
Legend: [R] required by task list, [G] grey area (in project header, implied by tasks)

### Build
- [x] [R] Task 0 — Project setup (venv, deps, .env, folders, git)
- [ ] [R] Task 1 — Ingestion: load PDFs/MD/FAQ/CSV, clean, extract metadata, store processed
- [ ] [R] Task 2 — Vector index: chunk, embed, store in ChromaDB, validate retrieval
- [ ] [R] Task 3 — Knowledge graph: extract entities/relations, load Neo4j, validate queries
- [ ] [R] Task 4 — Hybrid retrieval: vector + graph, merge, re-rank
- [ ] [R] Task 5 — LangGraph agent: web search, query tool, Python execution, multi-step reasoning
- [ ] [G] MCP-based integration (wrap one tool as an MCP server)
- [ ] [R] Task 6 — Guardrails: input validation, prompt-injection detection, PII masking, hallucination check, output filtering
- [ ] [R] Task 7 — FastAPI: /chat endpoint, sessions, structured responses with citations
- [ ] [R] Task 8 — Langfuse: tracing, latency, token usage, errors
- [ ] [G] Semantic caching
- [ ] [G] Model routing (Haiku vs Sonnet)

### Evaluate
- [ ] [R] Task 9 — Promptfoo suite, 30+ cases, regression run, failure analysis
- [ ] [R] Task 10 — RAGAS: faithfulness, answer relevance, context precision, context recall
- [ ] [R] Task 11 — Safety eval: adversarial prompts, failure rate, safety scorecard
- [ ] [R] Task 12 — Performance: latency, caching test, throughput

### Deliver
- [ ] [R] GitHub repository
- [ ] [R] Documentation: overview, architecture, solution approach, stepwise solution
- [ ] [R] Langfuse tracing screenshots
- [ ] [R] Promptfoo test suite + results
- [ ] [R] RAGAS evaluation report
- [ ] [R] Safety scorecard
- [ ] [R] Walkthrough: business problem, architecture, retrieval strategy, agent workflow,
      guardrails, evaluation results, observability insights, challenges/improvements

### Out of scope (do not build)
Frontend UI, enterprise auth, distributed infrastructure, model training.
