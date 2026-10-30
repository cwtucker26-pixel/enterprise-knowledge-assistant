"""Verify Task 1: document counts by type, a sample document's metadata, SQLite row counts."""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCUMENTS_PATH, SQLITE_PATH
from src.ingestion.metadata import CANONICAL_SYSTEMS

EXPECTED_DOCS = {"policy": 4, "incident_report": 5, "sop": 7, "technical_manual": 5, "faq": 11}
EXPECTED_ROWS = {"users": 46, "products": 7, "transactions": 200}


def main() -> int:
    problems = []
    docs = [json.loads(line) for line in open(DOCUMENTS_PATH, encoding="utf-8")]

    by_type = Counter(d["metadata"]["doc_type"] for d in docs)
    print(f"Documents: {len(docs)}")
    for doc_type, n in sorted(by_type.items()):
        print(f"  {doc_type:18} {n}")
    if dict(by_type) != EXPECTED_DOCS:
        problems.append(f"doc_type counts {dict(by_type)} != expected {EXPECTED_DOCS}")

    bad_refs = {r for d in docs for r in d["metadata"]["system_refs"] if r not in CANONICAL_SYSTEMS}
    if bad_refs:
        problems.append(f"non-canonical system names: {sorted(bad_refs)}")
    if any(not d["text"].strip() for d in docs):
        problems.append("a document has empty text")
    if len({d["doc_id"] for d in docs}) != len(docs):
        problems.append("duplicate doc_id")

    sample = next(d for d in docs if d["doc_id"] == "INC-204")
    print("\nSample metadata (INC-204):")
    print(json.dumps(sample["metadata"], indent=2))

    print("\nSQLite rows:")
    with sqlite3.connect(SQLITE_PATH) as conn:
        for table, expected in EXPECTED_ROWS.items():
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:14} {n}")
            if n != expected:
                problems.append(f"{table} has {n} rows, expected {expected}")

    if problems:
        print("\nFAIL:\n  " + "\n  ".join(problems))
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
