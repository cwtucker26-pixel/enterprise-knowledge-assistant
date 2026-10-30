"""Hybrid retrieval: vector search + graph expansion, merged, then re-ranked by a cross-encoder."""
import re

from langchain_core.documents import Document

from src.config import MAX_GRAPH_DOCS, MAX_GRAPH_FACTS, VECTOR_CANDIDATES
from src.retrieval import graph
from src.retrieval.rerank import Reranker, select_top
from src.retrieval.vector import get_embeddings, get_vector_store

TYPE_RANK = {"DEPENDS_ON": 0, "OWNS": 2, "MANAGES": 2, "RESOLVED_BY": 2, "RELATED_TO": 3}


def build_patterns(entities: list[dict]) -> list[tuple[dict, re.Pattern]]:
    """Case-insensitive; hyphens, spaces and underscores are interchangeable ("payment service" matches
    Payment-Service). Teams only match as "<team> team" because bare words like "security" are too common."""
    patterns = []
    for ent in entities:
        parts = [re.escape(p) for p in re.split(r"[-_\s]+", ent["name"]) if p]
        body = r"[-_\s]+".join(parts)
        if ent["label"] == "Team":
            body += r"\s+team"
        patterns.append((ent, re.compile(rf"(?<!\w){body}(?!\w)", re.I)))
    return patterns


def link_entities(question: str, patterns) -> list[dict]:
    return [ent for ent, pat in patterns if pat.search(question)]


def rank_graph_docs(rows: list[dict], hops: dict[str, int], limit: int) -> list[str]:
    """Closest document first (hop of its nearest mentioned entity); ties go to more mentions of it."""
    best: dict[str, tuple[int, int]] = {}
    for r in rows:
        h = hops[r["entity"]]
        cur = best.get(r["doc_id"])
        if cur is None or h < cur[0]:
            best[r["doc_id"]] = (h, r["count"])
        elif h == cur[0]:
            best[r["doc_id"]] = (h, cur[1] + r["count"])
    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0]))
    return [doc_id for doc_id, _ in ordered[:limit]]


def format_fact(f: dict) -> str:
    rel = f"RELATED_TO[{f['relation']}]" if f["type"] == "RELATED_TO" else f["type"]
    sources = ", ".join(sorted(f["sources"]))
    note = ""
    if f["type"] == "DEPENDS_ON":
        note = "; authoritative" if f["authoritative"] else "; NOT authoritative (incident report)"
    return f"{f['a']} {rel} {f['b']} (source: {sources}{note})"


def build_graph_facts(edges: list[dict], linked_keys: set, hops: dict[str, int], limit: int) -> list[str]:
    """Edges touching a linked entity first, then edges touching a 1-hop neighbour; DEPENDS_ON before the rest."""
    merged: dict[tuple, dict] = {}
    for e in edges:
        key = (e["a_key"], e["type"], e["relation"], e["b_key"], bool(e["authoritative"]))
        fact = merged.setdefault(key, {**e, "authoritative": bool(e["authoritative"]), "sources": set()})
        fact["sources"].add(e["source"])
    def sort_key(f):
        touches_linked = f["a_key"] in linked_keys or f["b_key"] in linked_keys
        auth_first = 0 if f["authoritative"] else 1
        return (0 if touches_linked else 1, TYPE_RANK[f["type"]], auth_first, f["a"], f["b"])
    return [format_fact(f) for f in sorted(merged.values(), key=sort_key)[:limit]]


class HybridRetriever:
    def __init__(self):
        self.store = get_vector_store(get_embeddings())
        self.driver = graph.make_driver()
        self.reranker = Reranker()
        self.patterns = build_patterns(graph.entity_names(self.driver))

    def close(self):
        self.driver.close()

    def _vector_candidates(self, question: str) -> list[Document]:
        results = self.store.similarity_search_with_score(question, k=VECTOR_CANDIDATES)
        return [Document(page_content=d.page_content, metadata={**d.metadata, "origin": "vector"}) for d, _ in results]

    def _chunks_for_docs(self, doc_ids: list[str]) -> list[Document]:
        if not doc_ids:
            return []
        got = self.store.get(where={"doc_id": {"$in": doc_ids}}, include=["documents", "metadatas"])
        return [Document(page_content=t, metadata={**m, "origin": "graph"}) for t, m in zip(got["documents"], got["metadatas"])]

    def retrieve(self, question: str, use_graph: bool = True) -> dict:
        """Returns {chunks, graph_facts, linked_entities, stats}. use_graph=False is vector + re-rank only."""
        vector_chunks = self._vector_candidates(question)
        linked, facts, graph_doc_ids, graph_chunks = [], [], [], []
        if use_graph:
            linked = link_entities(question, self.patterns)
            if linked:
                linked_keys = {e["key"] for e in linked}
                hops = graph.expand_entities(self.driver, sorted(linked_keys))
                graph_doc_ids = rank_graph_docs(
                    graph.documents_mentioning(self.driver, list(hops)), hops, MAX_GRAPH_DOCS
                )
                graph_chunks = self._chunks_for_docs(graph_doc_ids)
                near = [k for k, h in hops.items() if h <= 1]
                facts = build_graph_facts(graph.edges_touching(self.driver, near), linked_keys, hops, MAX_GRAPH_FACTS)

        pool: dict[str, Document] = {}
        for doc in vector_chunks + graph_chunks:
            cid = doc.metadata["chunk_id"]
            if cid in pool:
                pool[cid].metadata["origin"] = "both"
            else:
                pool[cid] = doc
        docs = list(pool.values())
        top = select_top(docs, self.reranker.score(question, docs))
        stats = {
            "vector_chunks": len(vector_chunks),
            "graph_docs": graph_doc_ids,
            "graph_chunks": len(graph_chunks),
            "pool_size": len(docs),
        }
        return {"chunks": top, "graph_facts": facts, "linked_entities": linked, "stats": stats}
