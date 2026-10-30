"""Neo4j access and read queries over the knowledge graph (reused by hybrid retrieval in Task 4)."""
from contextlib import contextmanager

from neo4j import GraphDatabase

from src.config import load_settings


@contextmanager
def get_driver():
    s = load_settings()
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
    try:
        yield driver
    finally:
        driver.close()


def run_query(driver, cypher: str, **params) -> list[dict]:
    records, _, _ = driver.execute_query(cypher, parameters_=params)
    return [r.data() for r in records]


def dependents_of(driver, system: str) -> list[dict]:
    """Systems that depend on `system` per the technical manuals, with owner team and lead."""
    return run_query(driver, """
        MATCH (s:System)-[d:DEPENDS_ON {authoritative: true}]->(:System {key: $name})
        OPTIONAL MATCH (t:Team)-[:OWNS]->(s)
        OPTIONAL MATCH (p:Person)-[:MANAGES]->(t)
        RETURN DISTINCT s.key AS system, t.key AS team, p.name AS lead, d.source AS source
        ORDER BY system""", name=system)


def incidents_for_system(driver, system: str) -> list[dict]:
    """Incidents involving `system`, the SOP that resolved each, and that SOP's owners."""
    return run_query(driver, """
        MATCH (i:Incident)-[:RELATED_TO]->(:System {key: $name})
        MATCH (i)-[:RESOLVED_BY]->(sop:SOP)
        OPTIONAL MATCH (p:Person)-[:RELATED_TO {relation: 'procedure_owner'}]->(sop)
        OPTIONAL MATCH (t:Team)-[:OWNS]->(sop)
        RETURN DISTINCT i.key AS incident, sop.key AS sop, sop.title AS sop_title,
               p.name AS procedure_owner, t.key AS owner_team
        ORDER BY incident""", name=system)


def conflicting_dependencies(driver) -> list[dict]:
    """Pairs of systems with DEPENDS_ON edges in both directions, with each edge's source."""
    return run_query(driver, """
        MATCH (a:System)-[e1:DEPENDS_ON]->(b:System), (b)-[e2:DEPENDS_ON]->(a)
        WHERE a.key < b.key
        RETURN a.key AS a, b.key AS b,
               e1.source AS a_to_b_source, e1.authoritative AS a_to_b_authoritative,
               e2.source AS b_to_a_source, e2.authoritative AS b_to_a_authoritative
        ORDER BY a, b""")


def unconfirmed_dependencies(driver) -> list[dict]:
    """Incident-only DEPENDS_ON edges that no manual confirms and no reverse edge contradicts."""
    return run_query(driver, """
        MATCH (a:System)-[e:DEPENDS_ON {authoritative: false}]->(b:System)
        WHERE NOT EXISTS { (a)-[:DEPENDS_ON {authoritative: true}]->(b) }
          AND NOT EXISTS { (b)-[:DEPENDS_ON]->(a) }
        RETURN a.key AS a, b.key AS b, e.source AS source ORDER BY source""")
