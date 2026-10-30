"""Verify Task 3: graph counts, Q2 and Q3 answered by Cypher, and the conflicting DEPENDS_ON edges."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.graph import (
    conflicting_dependencies,
    dependents_of,
    get_driver,
    incidents_for_system,
    run_query,
    unconfirmed_dependencies,
)

EXPECTED_Q2 = {
    ("Payment-Service", "Billing", "Priya Sharma"),
    ("UserProfile-API", "Customer Success", "Amara Okafor"),
    ("DataWarehouse", "Data Engineering", "Chen Wei"),
    ("APIGateway", "Security", "Natalia Voss"),
}
EXPECTED_Q3 = {
    ("INC-201", "SOP-01", "Marcus Lee"),
    ("INC-204", "SOP-17", "Priya Sharma"),
    ("INC-205", "SOP-22", "Natalia Voss"),
}
EXPECTED_CONFLICTS = {("APIGateway", "Auth-DB"), ("Auth-DB", "Payment-Service")}


def main() -> int:
    problems = []
    with get_driver() as driver:
        print("Nodes by label:")
        for r in run_query(driver, "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l"):
            print(f"  {r['l']:<10} {r['c']}")
        print("Relationships by type:")
        for r in run_query(driver, "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t"):
            print(f"  {r['t']:<12} {r['c']}")
        print("RELATED_TO by relation:")
        for r in run_query(driver, "MATCH ()-[r:RELATED_TO]->() RETURN r.relation AS rel, count(*) AS c ORDER BY rel"):
            print(f"  {r['rel']:<16} {r['c']}")

        print("\nQ2: systems that depend on Auth-DB (manuals), owner team and lead")
        rows = dependents_of(driver, "Auth-DB")
        for r in rows:
            print(f"  {r['system']:<16} {r['team']:<18} {r['lead']:<14} (source: {r['source']})")
        got = {(r["system"], r["team"], r["lead"]) for r in rows}
        print("  ->", "PASS" if got == EXPECTED_Q2 else "FAIL")
        if got != EXPECTED_Q2:
            problems.append(f"Q2 mismatch: {sorted(got ^ EXPECTED_Q2)}")

        print("\nQ3: incidents involving Auth-DB, resolving SOP, SOP owner")
        rows = incidents_for_system(driver, "Auth-DB")
        for r in rows:
            print(f"  {r['incident']}  {r['sop']} ({r['sop_title']})  procedure owner: {r['procedure_owner']}, team: {r['owner_team']}")
        got = {(r["incident"], r["sop"], r["procedure_owner"]) for r in rows}
        print("  ->", "PASS" if got == EXPECTED_Q3 else "FAIL")
        if got != EXPECTED_Q3:
            problems.append(f"Q3 mismatch: {sorted(got ^ EXPECTED_Q3)}")

        print("\nConflicting DEPENDS_ON edges (both directions exist):")
        rows = conflicting_dependencies(driver)
        for r in rows:
            print(f"  {r['a']} -> {r['b']}: {r['a_to_b_source']} (authoritative={r['a_to_b_authoritative']})")
            print(f"  {r['b']} -> {r['a']}: {r['b_to_a_source']} (authoritative={r['b_to_a_authoritative']})")
            print()
        got = {(r["a"], r["b"]) for r in rows}
        if got != EXPECTED_CONFLICTS:
            problems.append(f"conflict pairs {sorted(got)} != expected {sorted(EXPECTED_CONFLICTS)}")

        print("Incident-only DEPENDS_ON edges with no manual and no reverse edge:")
        for r in unconfirmed_dependencies(driver):
            print(f"  {r['a']} -> {r['b']}  ({r['source']})")

        orphans = run_query(
            driver, "MATCH (d:Document) WHERE NOT (d)-[:MENTIONS]->() RETURN d.key AS d, d.doc_type AS t ORDER BY d"
        )
        print(f"\nDocuments with no MENTIONS edges: {[o['d'] for o in orphans] or 'none'}")
        # A FAQ pair can legitimately name no entity (FAQ-07 is about the assistant itself);
        # a manual, SOP, incident or policy that links to nothing means extraction broke.
        bad = [o["d"] for o in orphans if o["t"] != "faq"]
        if bad:
            problems.append(f"non-FAQ documents with no MENTIONS: {bad}")

    print("\n" + ("PASS" if not problems else "FAIL:\n  " + "\n  ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
