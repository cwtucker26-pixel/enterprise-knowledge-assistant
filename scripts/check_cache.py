"""Offline verification of the semantic cache (no API calls, no Neo4j): the local bge model only."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api.cache import SemanticCache, is_cacheable, signature  # noqa: E402
from src.retrieval.hybrid import build_patterns  # noqa: E402
from src.retrieval.vector import get_embeddings  # noqa: E402

failures = []


def expect(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


ENTITIES = [{"label": "System", "key": k, "name": k} for k in ("Payment-Service", "Auth-DB", "APIGateway")] + \
           [{"label": "SOP", "key": k, "name": k} for k in ("SOP-17", "SOP-22")] + \
           [{"label": "Person", "key": "U0001", "name": "Priya Sharma"}, {"label": "Team", "key": "Billing", "name": "Billing"}]
PATTERNS = build_patterns(ENTITIES)
EMB = get_embeddings()


def cos(a, b):
    va, vb = (np.array(v) / np.linalg.norm(v) for v in EMB.embed_documents([a, b]))
    return float(va @ vb)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def cache(**kw):
    return SemanticCache(EMB.embed_documents, PATTERNS, **kw)


def main():
    print("Hits")
    c = cache()
    c.store("Who owns Payment-Service?", {"answer": "Billing"})
    for q in ("Who owns Payment-Service?", "Who owns Payment-Service", "who owns payment service?", "WHO OWNS PAYMENT-SERVICE??"):
        hit = c.lookup(q)
        expect(f"hit: {q!r}", hit is not None and hit.payload["answer"] == "Billing", f"{hit}")
    h = c.lookup("Who owns Payment-Service?")
    h.payload["answer"] = "tampered"
    expect("a returned payload is a copy (cannot corrupt the cache)", c.lookup("Who owns Payment-Service?").payload["answer"] == "Billing")

    print("Misses that an embedding threshold alone would NOT catch")
    pairs = [
        ("Does Auth-DB depend on Payment-Service?", "Does Payment-Service depend on Auth-DB?", "reversed direction"),
        ("Which systems depend on Auth-DB?", "Which systems does Auth-DB depend on?", "reversed direction, same entity"),
        ("Does Payment-Service depend on Auth-DB?", "Does Payment-Service not depend on Auth-DB?", "negation"),
        ("Who is the procedure owner of SOP-22?", "Who is the procedure owner of SOP-17?", "different SOP"),
        ("Who owns Payment-Service?", "Who owns Auth-DB?", "different system"),
        ("List the top 3 incidents", "List the top 5 incidents", "different number"),
    ]
    for stored, probe, why in pairs:
        cc = cache()
        cc.store(stored, {"answer": "A"})
        sim = cos(stored, probe)
        expect(f"miss ({why}, cosine {sim:.3f})", cc.lookup(probe) is None)
    sim = cos(*pairs[0][:2])
    expect("the reversed-direction pair really is above the threshold (so the signature is doing the work)", sim >= 0.95, f"{sim:.3f}")

    print("Signature")
    expect("entity aliases agree", signature("who owns payment service", PATTERNS) == signature("Who owns Payment-Service?", PATTERNS))
    expect("direction changes the signature", signature("Which systems depend on Auth-DB?", PATTERNS) != signature("Which systems does Auth-DB depend on?", PATTERNS))

    print("TTL, size, switches")
    clock = Clock()
    c = cache(ttl_s=100, clock=clock)
    c.store("Who owns Payment-Service?", {"answer": "A"})
    clock.t = 99
    expect("hit just before the TTL", c.lookup("Who owns Payment-Service?") is not None)
    clock.t = 101
    expect("miss after the TTL, and the entry is purged", c.lookup("Who owns Payment-Service?") is None and len(c) == 0)
    c = cache(max_entries=2)
    for q in ("Who owns Payment-Service?", "Who owns Auth-DB?", "Who owns APIGateway?"):
        c.store(q, {"answer": q})
    expect("oldest entry evicted at max_entries", len(c) == 2 and c.lookup("Who owns Payment-Service?") is None and c.lookup("Who owns APIGateway?") is not None)
    c = cache(enabled=False)
    c.store("Who owns Payment-Service?", {"answer": "A"})
    expect("disabled cache stores and serves nothing", len(c) == 0 and c.lookup("Who owns Payment-Service?") is None)
    c = cache(threshold=0.9999)
    c.store("Who owns Payment-Service?", {"answer": "A"})
    expect("threshold is enforced (0.9999 rejects a case/punctuation variant)", c.lookup("who owns payment service") is None)

    print("What may be cached")
    ok = NS(blocked=False, warning=False, rewritten_query=None, tools_used=["sql_query"], guard_results=[NS(action="none")], answer="Billing owns it.")
    expect("a normal answer is cacheable", is_cacheable(ok))
    for label, change in (("blocked", {"blocked": True}), ("warned", {"warning": True}), ("follow-up (rewritten)", {"rewritten_query": "x"}),
                          ("used web_search", {"tools_used": ["web_search"]}), ("a guard failed open", {"guard_results": [NS(action="skipped")]}),
                          ("empty answer", {"answer": "  "})):
        expect(f"not cacheable: {label}", not is_cacheable(NS(**{**ok.__dict__, **change})))

    print("\n" + ("PASS" if not failures else "FAIL: " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
