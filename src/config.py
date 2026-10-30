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
