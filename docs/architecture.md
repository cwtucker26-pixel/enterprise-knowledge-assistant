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

## Hybrid retrieval: graph facts carry relationships, chunks carry explanation

Retrieval merges vector candidates with chunks from graph-selected documents, re-ranks them with a
cross-encoder (`ms-marco-MiniLM-L-6-v2`) and keeps the top 8 (at most 2 per document). Alongside the
chunks, the answer prompt receives up to 20 *graph facts*: short lines such as
`Priya Sharma RELATED_TO[procedure_owner] SOP-17 (source: SOP-17)`, each with its source document and,
for `DEPENDS_ON`, an `authoritative` flag.

The finding from the Task 4 evaluation (6 questions, `eval/results/retrieval_hybrid.txt`): the re-ranker
drives chunk coverage (15/26 expected sources for vector@8, 18/26 with re-ranking, 19/26 hybrid), and the
graph adds little there. A cross-encoder ranks by semantic similarity to the question text, so passages
that hold *answer-bearing structured fields* lose: the SOP overview chunks that contain `Procedure Owner:`
scored between -6.7 and -3.3 for "who owns those SOPs?", while the incident chunks scored +2 to +3.6, and
filled six of the eight slots. Getting those documents into the candidate pool (graph document selection
by dependency/resolution path) was necessary but not sufficient.

The graph's real contribution is therefore the **facts**, which state precise relationships (owners,
dependents, which edges conflict) without depending on how a passage ranks. The two sources are
complementary: graph facts carry the relationships, chunks carry the explanation (procedures, root
causes, policy wording). Fact selection matters as much as chunk selection. With the first ranking,
the 20-fact cap filled with `emits_to` and incident edges, and the owner of DataWarehouse and the SOP
owners never reached the model. Ranking ownership facts (`OWNS`, `system_owner`, `procedure_owner`,
`MANAGES`) about entities on a `DEPENDS_ON`/`RESOLVED_BY` path ahead of other edges, with `emits_to`
last, made the Q2 and Q3 answers complete.

Remaining gap: the 20-fact cap is tight for hub entities such as Auth-DB. In the Q3 answer, the owning
team of SOP-22 (Security) was cut by the cap, although its procedure owner (Natalia Voss) is present.
Question-aware fact selection (for example boosting facts by the entity types the question mentions) is a
possible later improvement and was deliberately not built.
