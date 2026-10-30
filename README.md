# Enterprise Knowledge Assistant

Capstone project: a hybrid-RAG (vector + graph) agent with guardrails, over the ACME Corp dataset.
The full documentation (overview, architecture, results) is written in the final phase; see `CLAUDE.md`
for the project guide and `docs/architecture.md` for design notes.

## Run the API

```bash
source .venv/bin/activate          # Python 3.12 virtualenv with requirements.txt installed
python scripts/run_ingestion.py    # once: documents.jsonl + SQLite
python scripts/build_index.py      # once: Chroma vector index
python scripts/build_graph.py      # once: Neo4j graph (needs the .env keys)
uvicorn src.api.main:app --port 8000
```

- Interactive docs (try requests in the browser): http://localhost:8000/docs
- Health check: `curl localhost:8000/health`
- Ask a question:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "Who owns Payment-Service?"}'
# reuse the returned session_id for follow-ups:
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "What does it depend on?", "session_id": "<session_id from the first response>"}'
```

Every request is traced in Langfuse (keys in `.env`); the response carries an `X-Trace-Id` header to find it.

Startup loads the embedding model, the re-ranker and the guard models once (about 10 to 20 seconds).

**Limitations:** sessions are held in memory, so they are lost on restart, never expire, and are not shared
between workers: run a single worker (the default). A `session_id` acts as a bearer token; there is no
authentication (out of scope). A `/chat` call takes roughly 7 to 25 seconds because it runs retrieval, the
agent and the model-based guards.
