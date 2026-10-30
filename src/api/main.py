"""FastAPI app: POST /chat and GET /health.

Models, the retriever and the compiled agent are built once in the lifespan. Sessions live in a MemorySaver
checkpointer keyed by session_id (in memory only: lost on restart, never expired, not shared between workers).
Endpoints are plain `def`, so FastAPI runs them in a threadpool and the blocking agent never stalls the event loop.
"""
import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import build_agent_graph, final_answer, run_agent, turn_messages
from src.api.schemas import ChatRequest, ChatResponse, GuardResult, SourceRef
from src.guardrails.output import extract_citations, redact_secrets
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger("acme.api")


class RedactSecrets(logging.Filter):
    """Never write secrets to the server log, including inside exception messages and tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = redact_secrets(logging.Formatter().formatException(record.exc_info))
        return True


logger.addFilter(RedactSecrets())
WARNING_PREFIX = "⚠️"


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = HybridRetriever()
    app.state.retriever = retriever
    app.state.checkpointer = MemorySaver()
    app.state.agent = build_agent_graph(retriever=retriever, checkpointer=app.state.checkpointer)
    app.state.session_locks = defaultdict(threading.Lock)
    app.state.locks_guard = threading.Lock()
    yield
    retriever.close()


app = FastAPI(title="Enterprise Knowledge Assistant", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)  # details stay in the server log
    return JSONResponse(status_code=500, content={"detail": "Internal error"})


def session_lock(app: FastAPI, session_id: str) -> threading.Lock:
    """One lock per session, so two simultaneous requests for the same session can't interleave their history."""
    with app.state.locks_guard:
        return app.state.session_locks[session_id]


def build_response(state: dict, session_id: str, latency_ms: int) -> ChatResponse:
    answer = final_answer(state)
    turn = turn_messages(state)  # this turn only: with a checkpointer, state["messages"] holds the whole session
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        blocked=bool(state.get("blocked")),
        warning=answer.startswith(WARNING_PREFIX),
        rewritten_query=state.get("standalone_question") or None,
        citations=extract_citations(answer),
        sources=[SourceRef(chunk_id=d.metadata["chunk_id"], doc_id=d.metadata["doc_id"]) for d in state.get("chunks", [])],
        tools_used=[c["name"] for m in turn for c in getattr(m, "tool_calls", [])],
        guard_results=[GuardResult(**g) for g in state.get("guard_results", [])],
        latency_ms=latency_ms,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    app_state = request.app.state
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    with session_lock(request.app, session_id):
        started = time.monotonic()
        try:
            state = run_agent(app_state.agent, req.question, config)
        except Exception:
            # A turn that died half-way can leave a tool request without its result in the saved history,
            # which would make every later request in this session fail. Drop the session instead.
            try:
                app_state.checkpointer.delete_thread(session_id)
            except Exception:
                logger.exception("could not reset session after an error")
            raise
        latency_ms = int((time.monotonic() - started) * 1000)
    return build_response(state, session_id, latency_ms)


@app.get("/health")
def health(request: Request):
    app_state = request.app.state
    checks = {"agent": getattr(app_state, "agent", None) is not None}
    for name, probe in (
        ("chroma", lambda: bool(app_state.retriever.store.get(limit=1)["ids"])),
        ("neo4j", lambda: app_state.retriever.driver.verify_connectivity() or True),
    ):
        try:
            checks[name] = bool(probe())
        except Exception:
            checks[name] = False  # no details in the response
    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", "checks": checks})
