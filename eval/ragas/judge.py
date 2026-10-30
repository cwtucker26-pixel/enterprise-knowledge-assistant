"""RAGAS judge setup: an Anthropic LLM and the local bge embeddings, wrapped for RAGAS."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compat  # noqa: F401,E402  (must come before any ragas import)

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

from src.config import EMBEDDING_MODEL, load_settings  # noqa: E402


class AnthropicJudge(ChatAnthropic):
    """RAGAS sets `temperature` on the model it is given. The Anthropic models used here reject sampling
    parameters (HTTP 400, or a ValueError from langchain-anthropic for Sonnet 5.5), so attempts to set them are
    ignored and any that still reach the request are removed."""

    def __setattr__(self, name, value):
        if name in ("temperature", "top_k", "top_p"):
            return
        super().__setattr__(name, value)

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        for key in ("temperature", "top_k", "top_p"):
            payload.pop(key, None)
        extra = payload.get("extra_body")  # the library may also move sampling parameters in here
        if isinstance(extra, dict):
            for key in ("temperature", "top_k", "top_p"):
                extra.pop(key, None)
            if not extra:
                payload.pop("extra_body")
        return payload


def make_judge(model: str | None = None):
    """Returns (ragas_llm, ragas_embeddings). Default judge: the reasoning model (Sonnet)."""
    s = load_settings()
    llm = AnthropicJudge(model=model or s.reasoning_model, api_key=s.anthropic_api_key, max_tokens=4096)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)
