"""Task 3: build the Neo4j knowledge graph from documents.jsonl + users.csv (clear and reload)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCUMENTS_PATH
from src.ingestion.graph_extract import extract_graph, load_users
from src.retrieval.graph import get_driver, run_query

LABELS = ["Person", "Team", "System", "SOP", "Policy", "Incident", "Document"]  # fixed set: safe to put in Cypher


def load_nodes(driver, nodes):
    by_label = defaultdict(list)
    for n in nodes:
        by_label[n["label"]].append({"key": n["key"], "props": n["props"]})
    for label, rows in by_label.items():
        driver.execute_query(
            f"UNWIND $rows AS r MERGE (n:{label} {{key: r.key}}) SET n += r.props", parameters_={"rows": rows}
        )


def load_edges(driver, edges):
    # MERGE identity = (source) or (source, relation); other properties are SET afterwards.
    groups = defaultdict(list)
    for e in edges:
        groups[(e["type"], e["from"][0], e["to"][0], e["relation"] is not None)].append(
            {"from": e["from"][1], "to": e["to"][1], "source": e["source"], "relation": e["relation"], "props": e["props"]}
        )
    for (type_, from_label, to_label, has_relation), rows in groups.items():
        ident = "{source: r.source, relation: r.relation}" if has_relation else "{source: r.source}"
        driver.execute_query(
            f"UNWIND $rows AS r MATCH (a:{from_label} {{key: r.from}}) MATCH (b:{to_label} {{key: r.to}}) "
            f"MERGE (a)-[e:{type_} {ident}]->(b) SET e += r.props",
            parameters_={"rows": rows},
        )


def main():
    documents = [json.loads(line) for line in open(DOCUMENTS_PATH, encoding="utf-8")]
    nodes, edges, notes = extract_graph(documents, load_users())
    print(f"Extracted {len(nodes)} nodes and {len(edges)} edges")
    for note in notes:
        print("  note:", note)
    assert all(n["label"] in LABELS for n in nodes)

    with get_driver() as driver:
        existing = run_query(driver, "MATCH (n) RETURN count(n) AS n")[0]["n"]
        print(f"Clearing {existing} existing nodes from the graph")
        driver.execute_query("MATCH (n) DETACH DELETE n")
        for label in LABELS:
            driver.execute_query(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE")
        load_nodes(driver, nodes)
        load_edges(driver, edges)
        n_nodes = run_query(driver, "MATCH (n) RETURN count(n) AS n")[0]["n"]
        n_rels = run_query(driver, "MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
    print(f"Loaded graph: {n_nodes} nodes, {n_rels} relationships")
    if (n_nodes, n_rels) != (len(nodes), len(edges)):
        sys.exit(f"MISMATCH: expected {len(nodes)} nodes / {len(edges)} relationships")


if __name__ == "__main__":
    main()
