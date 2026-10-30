"""Deterministic graph extraction: documents.jsonl + users.csv -> nodes and edges (no database).

Every edge carries `source` (the doc_id, or "users.csv" for people/teams). Facts come from
templated headers and fixed sentence patterns, so plain regexes are enough (see docs/architecture.md).
Node identity is (label, key); the key property is stored as `key` on every node.
"""
import csv
import re

from src.config import RAW_DIR
from src.ingestion.metadata import CANONICAL_SYSTEMS, normalise_system


def node(label, key, **props):
    return {"label": label, "key": key, "props": props}


def edge(type_, src, dst, source, relation=None, **props):
    """src/dst are (label, key). `relation` is used for RELATED_TO to say what the link means."""
    return {"type": type_, "from": src, "to": dst, "source": source, "relation": relation, "props": props}


def header_field(text: str, name: str) -> str | None:
    m = re.search(rf"\*\*{name}:\*\*\s*(.+)", text)
    return m.group(1).strip() if m else None


def mention_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")


def load_users() -> list[dict]:
    with open(RAW_DIR / "structured" / "users.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_graph(documents: list[dict], users: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    nodes, edges, notes = {}, [], []

    def add_node(label, key, **props):
        nodes.setdefault((label, key), node(label, key, **props))

    # --- People and teams (users.csv). Emails are PII, so they are not copied into the graph. ---
    for u in users:
        add_node("Team", u["team"])
        add_node("Person", u["user_id"], name=u["name"], role=u["role"], active=u["active"] == "True")
        edges.append(edge("MEMBER_OF", ("Person", u["user_id"]), ("Team", u["team"]), "users.csv"))
        if u["role"] == "Team Lead":
            edges.append(edge("MANAGES", ("Person", u["user_id"]), ("Team", u["team"]), "users.csv"))
    person_id = {u["name"]: u["user_id"] for u in users}
    team_names = sorted({u["team"] for u in users})
    for name in CANONICAL_SYSTEMS:
        add_node("System", name)

    owners: dict[str, set] = {}  # system -> teams claiming ownership, to spot disagreements

    def owns_system(team, system, source):
        owners.setdefault(system, set()).add(team)
        edges.append(edge("OWNS", ("Team", team), ("System", system), source))

    # --- Per-document facts ---
    for doc in documents:
        doc_id, text, meta = doc["doc_id"], doc["text"], doc["metadata"]
        doc_type = meta["doc_type"]
        add_node("Document", doc_id, doc_type=doc_type, source_file=meta["source_file"])

        if doc_type == "technical_manual":
            system = normalise_system(re.search(r"# Technical Manual — (.+)", text).group(1))
            owns_system(header_field(text, "Owner Team"), system, doc_id)
            owner = header_field(text, "System Owner")
            edges.append(edge("RELATED_TO", ("Person", person_id[owner]), ("System", system), doc_id,
                              relation="system_owner"))
            for m in re.finditer(r"^([\w-]+) depends on \*\*([\w-]+)\*\*", text, re.M):
                a, b = normalise_system(m.group(1)), normalise_system(m.group(2))
                if a == b:
                    notes.append(f"{doc_id}: skipped self-dependency '{a} depends on {b}'")
                    continue
                edges.append(edge("DEPENDS_ON", ("System", a), ("System", b), doc_id, authoritative=True))
            m = re.search(r"downstream event emission to services\s+including \*\*([\w-]+)\*\* and \*\*([\w-]+)\*\*", text)
            for target in (m.groups() if m else ()):
                edges.append(edge("RELATED_TO", ("System", system), ("System", normalise_system(target)),
                                  doc_id, relation="emits_to"))

        elif doc_type == "incident_report":
            add_node("Incident", doc_id, date=header_field(text, "Date"), severity=header_field(text, "Severity"))
            affected = normalise_system(header_field(text, "System Affected"))
            owns_system(header_field(text, "Owner Team"), affected, doc_id)
            sop_id = header_field(text, "Referenced SOP").split(" — ")[0]
            edges.append(edge("RESOLVED_BY", ("Incident", doc_id), ("SOP", sop_id), doc_id))
            for system in meta["system_refs"]:
                relation = "affected" if system == affected else "involved"
                edges.append(edge("RELATED_TO", ("Incident", doc_id), ("System", system), doc_id, relation=relation))
            lead = header_field(text, "Incident Lead")
            edges.append(edge("RELATED_TO", ("Person", person_id[lead]), ("Incident", doc_id), doc_id, relation="lead"))
            m = re.search(r"failures in \*\*([\w-]+)\*\* which depends\s+on ([\w-]+) for", text)
            if m:  # incident wording may contradict the manuals, so these are not authoritative
                a, b = normalise_system(m.group(1)), normalise_system(m.group(2))
                edges.append(edge("DEPENDS_ON", ("System", a), ("System", b), doc_id, authoritative=False))

        elif doc_type == "sop":
            title = re.match(r"# SOP-\d+ — (.+)", text).group(1)
            add_node("SOP", doc_id, title=title)
            edges.append(edge("OWNS", ("Team", header_field(text, "Owner Team")), ("SOP", doc_id), doc_id))
            proc_owner = header_field(text, "Procedure Owner")
            edges.append(edge("RELATED_TO", ("Person", person_id[proc_owner]), ("SOP", doc_id), doc_id,
                              relation="procedure_owner"))

        elif doc_type == "policy":
            add_node("Policy", doc_id, title=text.split("\n")[0].strip())
            sentence = re.search(r"Owned by[^.]*\.", text)  # only some policies state an owner
            if sentence:
                team = next((t for t in team_names if t in sentence.group(0)), None)
                person = next((p for p in person_id if p in sentence.group(0)), None)
                if team:
                    edges.append(edge("OWNS", ("Team", team), ("Policy", doc_id), doc_id))
                if person:
                    edges.append(edge("RELATED_TO", ("Person", person_id[person]), ("Policy", doc_id), doc_id,
                                      relation="owner"))

    # SOP/Incident nodes referenced before their own file was read: make sure they all exist.
    for e in edges:
        for label, key in (e["from"], e["to"]):
            if (label, key) not in nodes:
                raise ValueError(f"edge from {e['source']} points at missing node {label}:{key}")

    for system, teams in owners.items():
        if len(teams) > 1:
            notes.append(f"OWNS disagreement for {system}: {sorted(teams)}")

    # --- Document -MENTIONS-> entity (exact-string, case-sensitive matching) ---
    patterns = []
    for (label, key), n in nodes.items():
        if label == "Document":
            continue
        name = n["props"].get("name", key)  # people are matched by full name, others by key
        patterns.append((label, key, mention_pattern(name)))
    team_pat = re.compile(rf"\b({'|'.join(map(re.escape, team_names))})\*{{0,2}}\s+team\b")
    owner_team_pat = re.compile(rf"Owner Team:\*\*\s*({'|'.join(map(re.escape, team_names))})\b")
    for doc in documents:
        text, doc_id = doc["text"], doc["doc_id"]
        for label, key, pat in patterns:
            if label == "Team":
                continue
            count = len(pat.findall(text))
            if count:
                edges.append(edge("MENTIONS", ("Document", doc_id), (label, key), doc_id, count=count))
        team_hits: dict[str, int] = {}
        for m in list(team_pat.finditer(text)) + list(owner_team_pat.finditer(text)):
            team_hits[m.group(1)] = team_hits.get(m.group(1), 0) + 1
        for team, count in team_hits.items():
            edges.append(edge("MENTIONS", ("Document", doc_id), ("Team", team), doc_id, count=count))

    return list(nodes.values()), edges, notes
