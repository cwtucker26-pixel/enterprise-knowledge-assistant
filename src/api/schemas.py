"""Request and response models for the API."""
from pydantic import BaseModel, Field, field_validator

from src.config import MAX_INPUT_CHARS


class ChatRequest(BaseModel):
    question: str = Field(description=f"The user's question (1 to {MAX_INPUT_CHARS} characters)")
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
                                   description="Continue an existing session; a UUID is generated if missing")
    use_cache: bool = Field(default=True, description="Set false to bypass the semantic answer cache for this request")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > MAX_INPUT_CHARS:
            raise ValueError(f"question must be at most {MAX_INPUT_CHARS} characters")
        return v


class SourceRef(BaseModel):
    chunk_id: str
    doc_id: str


class GuardResult(BaseModel):
    guard: str
    passed: bool
    action: str
    reason: str = ""


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    blocked: bool = Field(description="True if the input guard refused the question or the output was blocked")
    warning: bool = Field(description="True if the answer carries a guard warning")
    rewritten_query: str | None = Field(description="The standalone question used for retrieval (follow-ups only)")
    citations: list[str] = Field(description="Citations parsed from the answer")
    sources: list[SourceRef] = Field(description="Chunks retrieved for this question")
    tools_used: list[str] = Field(description="Tools called this turn, in order")
    guard_results: list[GuardResult]
    latency_ms: int
    cache_hit: bool = Field(default=False, description="True if the answer was replayed from the semantic cache (also the X-Cache header)")
