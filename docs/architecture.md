# Architecture

*(Work in progress: sections are added as each task lands.)*

## Knowledge graph extraction: deterministic, not LLM-based

The Neo4j graph is built by plain code (`src/ingestion/graph_extract.py`), not by asking an LLM to
read the documents and list entities and relationships. In this dataset every fact the graph needs
sits in a templated header (`**Owner Team:**`, `**Referenced SOP:**`), a CSV column (`users.csv`,
`doc_metadata.csv`) or a fixed sentence pattern ("X depends on **Y**"), so regexes extract it
exactly. That has three advantages over LLM extraction: the rebuild is reproducible (the same input
always gives the same graph, which the idempotent rebuild and the checks rely on); there is no cost
or latency at build time; and the graph can't contain an invented edge that looks authoritative.
Every relationship records its `source` document, and `DEPENDS_ON` edges from the technical manuals
are marked `authoritative: true`, while the incident reports' conflicting edges are kept with
`authoritative: false` so retrieval and evaluation can see the disagreement instead of having it
silently resolved at build time. The trade-off is brittleness: if the document templates change, the
regexes must change too, and facts that exist only in free prose (for example which team a policy
sentence holds responsible) are caught only when the wording is fixed. If a question later needs such
a fact, an LLM extraction pass for that gap can be added behind an explicit flag.
