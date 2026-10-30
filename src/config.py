"""Central config. All other modules read settings from here, never from os.environ directly."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Data locations (plain paths, no secrets, so scripts that need no API keys can import them)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.jsonl"
SQLITE_PATH = PROCESSED_DIR / "acme.db"

# Vector index (Task 2)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "acme_docs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_MIN_CHARS = 200  # sections shorter than this are merged into a neighbour
CHUNK_MAX_CHARS = 1000  # hard cap; longer sections are split on line boundaries
CHUNK_OVERLAP_CHARS = 100  # only used when a section is force-split
TOP_K = 5

# Hybrid retrieval (Task 4)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_CANDIDATES = 20  # chunks fetched by vector search before re-ranking
MAX_GRAPH_DOCS = 8  # documents pulled in via the graph (kept small so the graph's contribution is visible)
MAX_GRAPH_FACTS = 20
GRAPH_MAX_HOPS = 2
RERANK_TOP_K = 8
MAX_CHUNKS_PER_DOC = 2

# Agent (Task 5)
MAX_AGENT_STEPS = 6  # agent-node calls per question; the last one is made without tools
SQL_ROW_LIMIT = 50
SQL_TIMEOUT_S = 5
PYTHON_TIMEOUT_S = 10
WEB_MAX_RESULTS = 3
TOOL_OUTPUT_MAX_CHARS = 4000

# Semantic answer cache (Task 12). Disable with the environment variable CACHE_ENABLED=false (or per request with use_cache=false).
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")
CACHE_SIMILARITY = 0.95  # cosine threshold; a hit ALSO needs the same structural signature (see src/api/cache.py)
CACHE_TTL_S = 3600
CACHE_MAX_ENTRIES = 500

# Guardrails (Task 6) and sessions (Task 7)
MAX_INPUT_CHARS = 2000
GUARD_MAX_RETRIES = 1  # how many times a failed answer is sent back to the agent
MAX_HISTORY_TURNS = 4  # the agent (and the query rewriter) see the current turn plus up to 3 earlier ones
PROMPT_CANARY = "ACME-CANARY-7f3a9c41"  # unique marker in the system prompt; it appearing in an answer means a leak

REQUIRED_KEYS = [
    "ANTHROPIC_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "TAVILY_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    tavily_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str

    # Locked stack choices (not secrets)
    reasoning_model: str = "claude-sonnet-5-5"
    cheap_model: str = "claude-haiku-5-5"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"


def load_settings() -> Settings:
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k, "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required settings in .env: "
            + ", ".join(missing)
            + ". See .env.example for the full list."
        )
    values = {k.lower(): os.environ[k].strip() for k in REQUIRED_KEYS}
    return Settings(**values)
