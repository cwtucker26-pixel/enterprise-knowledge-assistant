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

## Dataset: ACME Corp (provided by the course; we do NOT generate data)
Internal IT-operations knowledge for a mid-size software/payments company. Files live in
`data/raw/`; `data/raw/raw_data_explanation.md` describes them. Everything is cross-linked by
system, team, person, and SOP.
- **PDFs (4):** POL-001 Acceptable Use, POL-002 Data Retention, POL-003 Information Security,
  POL-004 Remote Work (`pdfs/`)
- **Markdown (17):** incident reports INC-201..205, SOP-01..05/17/22, and technical manuals for
  Payment-Service, Auth-DB, UserProfile-API, DataWarehouse, APIGateway (`markdown/`)
- **FAQ:** `faq/FAQ.md` (11 Q&A pairs)
- **Structured (`structured/`):** `users.csv` (46 people), `products.csv` (7), `transactions.csv`
  (200 rows, all via Payment-Service) -> loaded into SQLite for the SQL tool
- **Metadata map:** `doc_metadata.csv` (filename, doc_type, dept, date, author, system_refs,
  sop_refs) and `manifest.json` drive metadata filtering and graph building. Names are cased
  inconsistently there (Auth-Db, Apigateway) -> normalise to Auth-DB, APIGateway, etc.
- **Graph:** no seed file; extract from documents + `doc_metadata.csv` + `users.csv`.
  - Entities: Person, Team, System, SOP, Policy, Incident
  - Relationships: `DEPENDS_ON` (System->System), `OWNS` (Team->System/Product),
    `MANAGES` (Person->Team), `MEMBER_OF` (Person->Team, from users.csv),
    `RESOLVED_BY` (Incident->SOP), `RELATED_TO` (anything else, e.g. Incident->System)

Source-of-truth rules and deliberate test material:
- **Manuals are the source of truth for `DEPENDS_ON`.** Incident reports sometimes invert the
  direction (INC-204: "Auth-DB depends on Payment-Service"; INC-205: "Auth-DB depends on
  APIGateway"), `manual_auth_db.md` says Auth-DB depends on itself, and the FAQ omits DataWarehouse
  from Auth-DB's dependents. Keep these conflicts: they test re-ranking and hallucination checks.
  Never "fix" the raw files.
- Notification-Service and ReportingPortal have no manual (owner is known only from incidents);
  InventoryEngine appears in POL-001 and a manual but has no owner.
- The incident/SOP/manual files are template-generated near-duplicates (re-ranking test).
- FAQ says Marcus Lee escalates to VP Engineering -> CTO, SOP-05 says Marcus Lee -> Natalia Voss
  -> VP Engineering (conflict). FAQ also says the router uses gpt-4o-mini/gpt-4o; our stack is
  Claude Haiku/Sonnet, so treat that FAQ answer as dataset text, not as our design.
- **PII:** the only fake PII is the 46 `@acme.internal` emails (plus names) in `users.csv`.
  There are no phone numbers or SSNs; the PII-masking tests must use emails and names, and
  any phone-number/SSN test cases will be written by us as adversarial inputs.
- Test questions with expected answers: `eval/test_questions.md`.

## Project structure
```
data/raw/{pdfs,markdown,faq,structured}/  data/raw/doc_metadata.csv  data/processed/
src/config.py
src/ingestion/      loaders, cleaning, chunking, metadata
src/retrieval/      vector.py, graph.py, hybrid.py, rerank.py
src/agent/          state.py, graph.py, tools.py
src/guardrails/     input.py, output.py
src/api/main.py
src/observability/
eval/{promptfoo,ragas,safety}/
scripts/            build_index.py, build_graph.py, ask.py
docs/               architecture.md, screenshots/
```

## Requirements checklist
Legend: [R] required by task list, [G] grey area (in project header, implied by tasks)

### Build
- [x] [R] Task 0 — Project setup (venv, deps, .env, folders, git)
- [x] [R] Task 1 — Ingestion: load PDFs/MD/FAQ/CSV, clean, extract metadata, store processed
- [x] [R] Task 2 — Vector index: chunk, embed, store in ChromaDB, validate retrieval
- [x] [R] Task 3 — Knowledge graph: extract entities/relations, load Neo4j, validate queries
- [x] [R] Task 4 — Hybrid retrieval: vector + graph, merge, re-rank
- [x] [R] Task 5 — LangGraph agent: web search, query tool, Python execution, multi-step reasoning
- [ ] [G] MCP-based integration (wrap one tool as an MCP server)
- [x] [R] Task 6 — Guardrails: input validation, prompt-injection detection, PII masking, hallucination check, output filtering
- [x] [R] Task 7 — FastAPI: /chat endpoint, sessions, structured responses with citations
- [x] [R] Task 8 — Langfuse: tracing, latency, token usage, errors
- [x] [G] Semantic caching
- [ ] [G] Model routing (Haiku vs Sonnet) — deliberately skipped, see eval/results/performance_report.md

### Evaluate
- [x] [R] Task 9 — Promptfoo suite, 30+ cases, regression run, failure analysis
- [x] [R] Task 10 — RAGAS: faithfulness, answer relevance, context precision, context recall
- [x] [R] Task 11 — Safety eval: adversarial prompts, failure rate, safety scorecard
- [x] [R] Task 12 — Performance: latency, caching test, throughput

### Deliver
- [ ] [R] GitHub repository
- [x] [R] Documentation: overview, architecture, solution approach, stepwise solution
- [x] [R] Langfuse tracing screenshots
- [x] [R] Promptfoo test suite + results
- [x] [R] RAGAS evaluation report
- [x] [R] Safety scorecard
- [x] [R] Walkthrough: business problem, architecture, retrieval strategy, agent workflow,
      guardrails, evaluation results, observability insights, challenges/improvements

### Out of scope (do not build)
Frontend UI, enterprise auth, distributed infrastructure, model training.
