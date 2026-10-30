"""Cross-encoder re-ranking: scores each (question, passage) pair together, then picks the top k."""
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import MAX_CHUNKS_PER_DOC, RERANK_MODEL, RERANK_TOP_K


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def score(self, question: str, docs: list[Document]) -> list[float]:
        """Raw logits: only comparable between passages for the same question."""
        if not docs:
            return []
        return [float(s) for s in self.model.predict([(question, d.page_content) for d in docs])]


def select_top(
    docs: list[Document], scores: list[float], k: int = RERANK_TOP_K, max_per_doc: int = MAX_CHUNKS_PER_DOC
) -> list[Document]:
    """Best-first, keeping at most `max_per_doc` chunks from any one document. Stores the score in metadata."""
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    chosen, per_doc = [], {}
    for doc, score in ranked:
        doc_id = doc.metadata["doc_id"]
        if per_doc.get(doc_id, 0) >= max_per_doc:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        chosen.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": score}))
        if len(chosen) == k:
            break
    return chosen
