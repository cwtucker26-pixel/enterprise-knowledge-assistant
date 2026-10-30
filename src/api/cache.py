"""Semantic answer cache for FIRST-TURN questions.

A hit needs BOTH:
  1. embedding similarity >= threshold (cosine of the bge vectors), and
  2. the same structural signature: the ordered entity mentions (canonical names, so "payment service" and
     "Payment-Service" agree), plus numbers, negations, question words and the small words that carry direction
     ("on", "of", "by", "does", ...).

Why (2): sentence embeddings barely see word order. Measured with this project's model, "Does Auth-DB depend on
Payment-Service?" vs the reversed question has cosine 0.996, HIGHER than genuine paraphrases (0.92 to 0.97), so no
threshold can separate them and an embedding-only cache would serve the wrong answer. The signature makes the cache
conservative: in practice it hits on the same wording up to case, punctuation and hyphenation, and on paraphrases
that keep the entities and direction words in the same order.

Only answers that are safe to replay are stored (see `is_cacheable`). Nothing is stored or served for follow-ups.
"""
import copy
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

# Words kept in the signature because changing or reordering them can change the answer.
KEEP_WORDS = {
    "who", "whom", "whose", "what", "which", "how", "when", "where", "why",
    "does", "do", "did", "is", "are", "was", "were", "can", "should", "will",
    "not", "no", "never", "without", "only", "except",
    "on", "of", "by", "from", "to", "for", "in", "at", "before", "after", "than", "between", "with", "if",
    "all", "any", "each", "every", "most", "least", "first", "last", "many", "much",
}


def signature(question: str, patterns) -> tuple:
    """Ordered structural fingerprint of a question. `patterns` is HybridRetriever.patterns:
    [({"key": ..., "name": ...}, compiled_regex)]."""
    spans = []
    for ent, pat in patterns:
        spans += [(m.start(), m.end(), ent["key"]) for m in pat.finditer(question)]
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    tokens, pos = [], 0
    for start, end, key in spans:
        if start < pos:  # overlaps an entity already taken
            continue
        tokens += _words(question[pos:start])
        tokens.append("E:" + key)
        pos = end
    tokens += _words(question[pos:])
    return tuple(t for t in tokens if t.startswith("E:") or t in KEEP_WORDS or any(c.isdigit() for c in t) or t.endswith("n't"))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def is_cacheable(response) -> bool:
    """Safe to replay: a normal, first-turn, unwarned answer that does not depend on live web data."""
    if response.blocked or response.warning or response.rewritten_query:
        return False
    if "web_search" in response.tools_used:  # time-dependent
        return False
    if any(g.action == "skipped" for g in response.guard_results):  # a model guard failed open: unverified
        return False
    return bool(response.answer.strip())


@dataclass
class CacheHit:
    payload: dict
    similarity: float
    matched_question: str


class SemanticCache:
    def __init__(self, embed, patterns, *, threshold: float = 0.95, ttl_s: float = 3600,
                 max_entries: int = 500, enabled: bool = True, clock=time.monotonic):
        self.embed, self.patterns = embed, patterns
        self.threshold, self.ttl_s, self.max_entries, self.enabled, self.clock = threshold, ttl_s, max_entries, enabled, clock
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def _vector(self, text: str) -> np.ndarray:
        v = np.asarray(self.embed([text])[0], dtype=float)
        return v / np.linalg.norm(v)

    def _purge(self) -> None:
        now = self.clock()
        self._entries = [e for e in self._entries if now - e["stored_at"] <= self.ttl_s]

    def lookup(self, question: str) -> CacheHit | None:
        if not self.enabled:
            return None
        vec, sig = self._vector(question), signature(question, self.patterns)
        with self._lock:
            self._purge()
            best = None
            for e in self._entries:
                sim = float(vec @ e["vec"])
                if sim >= self.threshold and e["sig"] == sig and (best is None or sim > best[0]):
                    best = (sim, e)
            if best is None:
                self.misses += 1
                return None
            self.hits += 1
            return CacheHit(copy.deepcopy(best[1]["payload"]), best[0], best[1]["question"])

    def store(self, question: str, payload: dict) -> None:
        if not self.enabled:
            return
        entry = {"question": question, "vec": self._vector(question), "sig": signature(question, self.patterns),
                 "payload": copy.deepcopy(payload), "stored_at": self.clock()}
        with self._lock:
            self._purge()
            self._entries.append(entry)
            del self._entries[:-self.max_entries]  # oldest first out

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)
