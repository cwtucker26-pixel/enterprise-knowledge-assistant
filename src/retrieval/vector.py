"""Vector store: local bge embeddings + persistent Chroma, via LangChain wrappers."""
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    BGE_QUERY_PREFIX,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    # bge wants a prefix on queries only (never on documents); set once here so callers can't forget.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": BGE_QUERY_PREFIX},
    )


def get_vector_store(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Delete the collection if it exists, so the next get_vector_store() starts empty."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)


def search(store: Chroma, query: str, k: int = TOP_K, where: dict | None = None):
    """Returns [(Document, cosine_distance)]; lower distance = closer."""
    return store.similarity_search_with_score(query, k=k, filter=where)
