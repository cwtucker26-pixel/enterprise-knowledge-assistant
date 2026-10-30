"""Langfuse tracing for /chat: one trace per request, session = the chat session, nodes as spans.

PII: every input/output/metadata value is passed through `mask_langfuse` before export, so raw emails and
phone numbers (and secrets) never reach Langfuse, whatever the graph put in the trace.
"""
import json
import time
from contextlib import ExitStack

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from src.guardrails.output import redact_secrets
from src.guardrails.pii import mask_pii

META_VALUE_MAX = 150  # Langfuse drops propagated metadata values over 200 characters (measured after encoding)


def _scrub(text: str) -> str:
    return redact_secrets(mask_pii(text)[0])


def mask_langfuse(data=None, **_):
    """Langfuse `mask` hook: recursively mask PII and secrets in anything about to be exported."""
    if data is None or isinstance(data, (bool, int, float)):
        return data
    if isinstance(data, str):
        return _scrub(data)
    if isinstance(data, dict):
        return {k: mask_langfuse(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_langfuse(v) for v in data]
    try:  # LangChain messages and other objects: convert to plain data first, then mask
        plain = json.loads(json.dumps(data, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else repr(o)))
        return mask_langfuse(plain)
    except Exception:
        return _scrub(repr(data))


def init_langfuse(settings) -> Langfuse:
    return Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host, mask=mask_langfuse)


def _short(value) -> str:
    return str(value)[:META_VALUE_MAX]


class ChatTrace:
    """Context manager around one /chat request: opens the root span, hands out the LangChain callback handler,
    and, once the result is known, attaches tags and metadata (guard results, tools, rewritten query) to the trace."""

    def __init__(self, langfuse: Langfuse, session_id: str, question: str):
        self.lf, self.session_id, self.question = langfuse, session_id, question
        self.trace_id = langfuse.create_trace_id()
        self.handler = None
        self._stack = ExitStack()
        self._root = None
        self._started = 0.0

    def __enter__(self):
        self._started = time.monotonic()
        self._root = self._stack.enter_context(self.lf.start_as_current_observation(
            name="chat", as_type="span", trace_context={"trace_id": self.trace_id},
            input={"question": self.question, "session_id": self.session_id}))
        self._stack.enter_context(propagate_attributes(session_id=self.session_id, trace_name="chat", tags=["chat"]))
        self.handler = CallbackHandler()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._stack.__exit__(exc_type, exc, tb)

    def record_result(self, response) -> None:
        """Called inside the `with` block after the response is built."""
        guards = [g.model_dump() for g in response.guard_results]
        tags = ["chat"]
        if response.blocked:
            tags.append("blocked")
        if response.warning:
            tags.append("warned")
        if response.rewritten_query:
            tags.append("follow-up")
        tags += [f"tool:{name}" for name in dict.fromkeys(response.tools_used)]
        flagged = [g["guard"] for g in guards if not g["passed"] or g["action"] in ("flagged", "warned")]
        # Entering propagate_attributes again while the root span is current sets the attributes on the root
        # span, which is how values only known after the run end up on the trace.
        with propagate_attributes(session_id=self.session_id, trace_name="chat", tags=tags, metadata={
            "blocked": _short(response.blocked), "warning": _short(response.warning),
            "rewritten_query": _short(response.rewritten_query or ""),
            "tools_used": _short(",".join(response.tools_used)),
            "guards_flagged": _short(",".join(flagged)),
        }):
            self._root.update(
                output={"answer": response.answer, "citations": response.citations, "blocked": response.blocked,
                        "warning": response.warning, "rewritten_query": response.rewritten_query,
                        "tools_used": response.tools_used, "latency_ms": response.latency_ms},
                metadata={"guard_results": guards, "sources": [s.model_dump() for s in response.sources]})

    def record_error(self, exc: Exception) -> None:
        """Mark the trace as failed. Only the exception type is recorded, never its message."""
        with propagate_attributes(session_id=self.session_id, trace_name="chat", tags=["chat", "error"]):
            self._root.update(level="ERROR", status_message=type(exc).__name__)
