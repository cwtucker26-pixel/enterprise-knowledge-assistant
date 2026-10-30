"""Hybrid retrieval: vector search + graph expansion, merged, then re-ranked by a cross-encoder."""
import re

from langchain_core.documents import Document

from src.config import MAX_GRAPH_DOCS, MAX_GRAPH_FACTS, VECTOR_CANDIDATES
from src.retrieval import graph
from src.retrieval.rerank import Reranker, select_top
from src.retrieval.vector import get_embeddings, get_vector_store


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


PRIORITY_VIA = {"DEPENDS_ON", "RESOLVED_BY"}  # entities reached through these edges are on a dependency/resolution path


def select_graph_docs(expanded: dict, subjects: dict, mention_rows: list[dict], linked_keys: set, limit: int) -> list[str]:
    """Pick up to `limit` documents, in three passes:
    1. the document about each linked entity (its manual, SOP, ...);
    2. documents about entities reached via DEPENDS_ON / RESOLVED_BY, nearest first, whatever the hop
       (so a SOP two hops away is not crowded out by the many documents that merely mention a hub);
    3. documents that mention expanded entities: nearest entity first, ties to more mentions.
    """
    chosen: list[str] = []

    def add(doc_id):
        if doc_id not in chosen and len(chosen) < limit:
            chosen.append(doc_id)

    for key in sorted(linked_keys):
        if key in subjects:
            add(subjects[key])
    priority = sorted(
        (k for k, v in expanded.items() if k not in linked_keys and v["via"] & PRIORITY_VIA and k in subjects),
        key=lambda k: (expanded[k]["hops"], k),
    )
    for key in priority:
        add(subjects[key])

    best: dict[str, tuple[int, int]] = {}
    for r in mention_rows:
        h = expanded[r["entity"]]["hops"]
        cur = best.get(r["doc_id"])
        if cur is None or h < cur[0]:
            best[r["doc_id"]] = (h, r["count"])
        elif h == cur[0]:
            best[r["doc_id"]] = (h, cur[1] + r["count"])
    for doc_id, _ in sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0])):
        add(doc_id)
    return chosen


def format_fact(f: dict) -> str:
    rel = f"RELATED_TO[{f['relation']}]" if f["type"] == "RELATED_TO" else f["type"]
    sources = ", ".join(sorted(f["sources"]))
    note = ""
    if f["type"] == "DEPENDS_ON":
        note = "; authoritative" if f["authoritative"] else "; NOT authoritative (incident report)"
    return f"{f['a']} {rel} {f['b']} (source: {sources}{note})"


PERSON_RELATIONS = {"system_owner", "procedure_owner"}


def fact_tier(f: dict, linked_keys: set, path_keys: set, conflicts: set) -> int:
    """Lower tier = earlier in the facts list. Ownership facts about entities on a dependency/resolution
    path (and the linked entities) come right after the authoritative dependencies; emits_to comes last."""
    subject = linked_keys | path_keys
    if f["type"] == "DEPENDS_ON":
        touches_linked = f["a_key"] in linked_keys or f["b_key"] in linked_keys
        if f["authoritative"]:
            return 0 if touches_linked else 5
        return 2 if (f["a_key"], f["b_key"]) in conflicts and touches_linked else 5
    if f["type"] == "RELATED_TO" and f["relation"] in PERSON_RELATIONS and f["b_key"] in subject:
        return 1
    if f["type"] == "OWNS" and f["b_key"] in subject:
        return 3
    if f["type"] == "MANAGES" and f["b_key"] in f["owner_teams"]:
        return 4
    if f["type"] == "RELATED_TO" and f["relation"] == "emits_to":
        return 7
    return 6


def build_graph_facts(edges: list[dict], linked_keys: set, expanded: dict, subjects: dict, limit: int) -> list[str]:
    path_keys = {k for k, v in expanded.items() if k not in linked_keys and v["via"] & PRIORITY_VIA}
    merged: dict[tuple, dict] = {}
    for e in edges:
        key = (e["a_key"], e["type"], e["relation"], e["b_key"], bool(e["authoritative"]))
        fact = merged.setdefault(key, {**e, "authoritative": bool(e["authoritative"]), "sources": set()})
        fact["sources"].add(e["source"])
    facts = list(merged.values())
    # A non-authoritative edge whose reverse is authoritative is a conflict worth showing.
    conflicts = {(f["a_key"], f["b_key"]) for f in facts if f["type"] == "DEPENDS_ON" and not f["authoritative"]
                 and any(g["type"] == "DEPENDS_ON" and g["authoritative"] and (g["a_key"], g["b_key"]) == (f["b_key"], f["a_key"])
                         for g in facts)}
    subject = linked_keys | path_keys
    owner_teams = {f["a_key"] for f in facts if f["type"] == "OWNS" and f["b_key"] in subject}
    for f in facts:
        f["owner_teams"] = owner_teams

    def sort_key(f):
        tier = fact_tier(f, linked_keys, path_keys, conflicts)
        # Inside the ownership tiers: path entities before the linked entity's own owner, entities
        # with a document of their own first, then nearest, then by name.
        target = f["b_key"]
        entity_rank = (target in linked_keys, target not in subjects, expanded.get(target, {"hops": 9})["hops"], f["b"])
        return (tier, entity_rank if tier in (1, 3) else (), 0 if f["authoritative"] else 1, f["a"], f["b"])

    return [format_fact(f) for f in sorted(facts, key=sort_key)[:limit]]


class HybridRetriever:
    def __init__(self):
        self.embeddings = get_embeddings()  # shared with the semantic answer cache
        self.store = get_vector_store(self.embeddings)
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
                expanded = graph.expand_entities(self.driver, sorted(linked_keys))
                hops = {k: v["hops"] for k, v in expanded.items()}
                subjects = graph.subject_docs(self.driver, list(expanded))
                graph_doc_ids = select_graph_docs(
                    expanded,
                    subjects,
                    graph.documents_mentioning(self.driver, list(expanded)),
                    linked_keys,
                    MAX_GRAPH_DOCS,
                )
                graph_chunks = self._chunks_for_docs(graph_doc_ids)
                path = [k for k, v in expanded.items() if v["via"] & PRIORITY_VIA]
                near = sorted({k for k, h in hops.items() if h <= 1} | set(path))
                facts = build_graph_facts(
                    graph.edges_touching(self.driver, near), linked_keys, expanded, subjects, MAX_GRAPH_FACTS
                )

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
