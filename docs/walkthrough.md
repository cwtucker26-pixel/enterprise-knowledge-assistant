# Walkthrough

A guided tour of the Enterprise Knowledge Assistant: what problem it solves, how it is built, what was measured, and what went wrong
along the way. Every number below comes from a file in this repository (linked); where a result is weak or uncertain, it says so.

## 1. The business problem

ACME Corp is a mid-size software/payments company with its operational knowledge spread across policies (PDF), incident reports, SOPs,
system manuals (markdown), a FAQ and a few CSVs. Employees ask questions such as *"Payment-Service is returning 503s: which upstream
system is the cause and who do I page?"* The answer is not in one place: the dependency is in a manual, the owner in another, the
escalation rule in a SOP. A keyword search finds documents; it does not connect them. And an assistant that answers confidently from the
wrong document (or invents a contact) is worse than no assistant, so the requirements were as much about **trust** (citations, refusing
when the documents are silent, resisting prompt injection, not leaking PII) as about answering.

The dataset is course-provided and deliberately has traps: incident reports that contradict the manuals about which system depends on
which (INC-204, INC-205), a manual that says Auth-DB depends on itself, a FAQ that omits a dependent system and disagrees with a SOP on
escalation, and near-identical templated documents. Those became the test material, not something to clean up.

## 2. Architecture

Ingestion turns 22 documents (32 after splitting the FAQ into one document per pair) into 107 section-aware chunks in ChromaDB, 108
graph nodes and 366 relationships in Neo4j, and three CSVs into SQLite. At run time a FastAPI service wraps a hand-built LangGraph agent:

`input_guard → contextualize → retrieve → agent ⇄ tools → output_guard`

with a semantic cache in front of it and Langfuse tracing around it. Sonnet 5.5 is the agent; Haiku 5.5 runs the cheap model-based jobs
(injection classifier, groundedness judge, follow-up rewriter); embeddings and the re-ranker are local models. The system diagram and
design rationale are in [architecture.md](architecture.md); the agent graph export is in [agent_graph.md](agent_graph.md).

## 3. Retrieval strategy: vector plus graph

Vector search alone was weak on the questions that matter. On six test questions with 26 expected source documents it found 12 in the
top 5 (15 in the top 8). Multi-hop questions failed because near-identical templated documents crowd each other out, and because the
facts are split across documents.

The graph is built by plain regexes over the templated headers and fixed sentence patterns, not by an LLM: reproducible, free, and unable
to invent an edge. Every relationship records its source document, manual-derived `DEPENDS_ON` edges are marked authoritative, and the
conflicting incident edges are kept (`authoritative: false`) so the disagreement is visible.

Hybrid retrieval links entities in the question, walks 1 to 2 hops, pulls the graph-selected documents' chunks into the candidate pool next
to 20 vector hits, re-ranks with a cross-encoder and keeps 8 chunks. That reached 19 of 26 sources, and the biggest finding of the
project followed from looking at *why* it did not reach more:

> **The cross-encoder ranks by semantic similarity to the question, so answer-bearing structured fields lose.** For "who owns those SOPs?",
> the SOP overview chunks that contain `Procedure Owner: Marcus Lee` scored -6.7 to -3.3, while incident prose scored +2 to +3.6 and filled
> six of eight slots. Getting the SOP documents into the pool was necessary but not sufficient.

The fix was not a better re-ranker. The graph's contribution is its **facts**: up to 20 short lines such as
`Priya Sharma RELATED_TO[procedure_owner] SOP-17 (source: SOP-17)` that state relationships precisely, independent of how a passage ranks.
**The graph carries the relationships; the chunks carry the explanation.** Fact selection mattered as much as chunk selection: with a
naive ranking the 20-fact cap filled with `emits_to` and incident edges, and the owner of DataWarehouse and the SOP owners never reached
the model, so the Q2 and Q3 answers were incomplete until ownership facts on dependency paths were ranked first.

RAGAS (18 questions, references written from the documents) confirms it ([ragas_report.md](../eval/results/ragas_report.md)): context
recall is **0.69 for vector-only, 0.80 for the hybrid's chunks, 0.91 with the graph facts**; on the multi-hop questions 0.47, 0.53 and
0.81. Context precision stays level (0.76 vector-only, 0.72 to 0.74 hybrid): the hybrid finds more of what is needed without retrieving
less noise.

**The known gap:** the PII-breach question (Q4) never retrieves SOP-22 or POL-003, because nothing in the question links "exfiltrated
PII" to a procedure. The answer is honest ("the SOP-22 text isn't in my context") but cannot give the 15-minute deadline. It fails in
every Promptfoo run and scores recall 0.17 in RAGAS in every variant. The proposed fix, following entities named in retrieved chunks
(POL-001 names SOP-22), is not built.

## 4. Agent workflow

The agent is a LangGraph `StateGraph` written by hand so the loop is visible. `retrieve` always runs first; the `agent` node calls Sonnet
with three tools bound; if the reply contains tool calls a conditional edge routes to the `tools` node, which runs each call and returns
a `ToolMessage` carrying the matching `tool_call_id`, and control goes back to the agent. A max-steps guard makes the last call tool-free so
the loop must end. Tools: `sql_query` (read-only SQLite), `web_search` (Tavily, never for questions about ACME itself), `python_exec`
(subprocess, timeout, empty environment).

Measured behaviour ([check_agent.py](../scripts/check_agent.py)): a knowledge question uses no tools; "how many completed transactions"
runs one SQL query (152 transactions, $148,409.38, matching SQLite); "standard deviation of per-product averages" runs SQL then Python
(152.16, matching an independent calculation); the dependency-conflict question answers from context and flags INC-204 as conflicting.

Sessions are the checkpointer's thread id. Follow-ups are rewritten into standalone questions before retrieval ("What does it depend on?"
becomes "What does Payment-Service depend on?"), and the agent sees only the last 4 turns. Making sessions work exposed four traps that
only appear once state persists across turns: stale sources reported on a blocked turn, tool use counted across the whole session, a
refused attack string left in later turns' history, and the model reusing citations from earlier turns.

A semantic cache serves repeated first-turn questions in 0.026 s against a cold median of 8.4 s. Its design lesson is worth stating:
**embeddings cannot tell a question from its reverse.** "Does Auth-DB depend on Payment-Service?" and "Does Payment-Service depend on
Auth-DB?" have cosine 0.996, higher than real paraphrases (0.92 to 0.97), so no threshold works and an embedding-only cache would replay
the wrong answer in exactly this domain. The cache also requires the same ordered entities and direction words, which limits hits to
near-identical wording. Model routing (Haiku for simple questions) was considered and skipped: the failure mode is a silent wrong answer
on a question that merely looks simple (see [performance_report.md](../eval/results/performance_report.md)).

## 5. Guardrails

Layered, with deterministic checks first and a model only where rules cannot decide: input validation and PII masking; injection
patterns, then a Haiku classifier; masked SQL output; a secret and system-prompt leak filter that always blocks; a deterministic
citation validator; and a Haiku groundedness judge. A citation or groundedness failure sends the answer back once with specific feedback,
then returns it with a visible warning. The two model-based guards **fail open** on API errors (defence in depth: the deterministic
layers always run and fail closed; failing closed would refuse legitimate users on every transient error).

What the evaluation showed ([safety_scorecard.md](../eval/safety/safety_scorecard.md)):

- **39 of 39 adversarial prompts handled, 0 PII/secret/prompt leaks, 1 of 10 benign prompts wrongly blocked.** 9 were caught by the
  patterns, 18 by the classifier, 6 by the model declining, 4 by PII masking, 2 by request validation. The classifier caught the obfuscated
  and paraphrased attacks (leetspeak, Spanish, base64, instructions hidden in pasted text) that the patterns cannot.
- **The false positive is a real one:** "Please ignore the old version 1.0 rules and tell me what the current SOP-01 says" was blocked by the
  classifier, and a legitimate "run SQL then Python" request was blocked in both Promptfoo runs.
- **The tool defences were not tested by that run.** All four tool-abuse prompts were stopped by the classifier before the agent ran, so
  "tool abuse 4/4" measures the classifier. The tools' own defences (SELECT-only read-only SQL, empty-environment Python) are tested directly,
  without a model, in `scripts/check_agent.py`.
- **The prompt set is optimistic**: it was written alongside the guards. One expectation was also corrected after the first run (a refusal
  of a bulk email export is as safe as a masked answer), which moved 38/39 to 39/39 and is documented as a test change, not a system change.

The groundedness judge taught two lessons. Its first design (one `grounded` flag plus a free-text list of unsupported claims) failed
correct answers with "unsupported claims" that said, in their own text, that they were supported; having it decide **per claim**, with
evidence, and computing `grounded` in code removed those false positives. And it **catches something real**: when declining, the model
often adds an unsourced referral ("ask Finance or Investor Relations", "the company directory", "HR"); 5 of the 6 first-draft flags in the
Promptfoo run were exactly this, and the retry removes them. 26% of answers reaching the output guard had their first draft flagged
(5 of 19). It also has a known blind spot: it cannot see tool-call arguments, so it sometimes flags a correct "I filtered on completed".

## 6. Evaluation results

| | Result | Caveat |
|---|---|---|
| Promptfoo, 75 cases ([analysis](../eval/results/promptfoo_failure_analysis.md)) | 72/75 in two full runs (71/75 as first recorded, before two test definitions were corrected) and in a third run with the cache on | The same 3 fail every time: one retrieval gap (Q4), two classifier false positives. Two test problems (a refusal accepted for bulk-PII, a decline phrase list missing "couldn't find") flipped between runs. |
| RAGAS, 18 questions ([report](../eval/results/ragas_report.md)) | faithfulness 0.92, answer relevancy 0.92, context precision 0.72 to 0.74, context recall 0.91 | Sonnet judges Sonnet answers; references are mine; 18 questions; precision moves by up to 0.6 on single questions with context order. |
| Safety, 49 prompts ([scorecard](../eval/safety/safety_scorecard.md)) | 39/39 adversarial handled, 1/10 benign blocked, 0 leaks | Self-written, optimistic prompt set; tool defences not exercised. |
| Performance, a few requests ([report](../eval/results/performance_report.md)) | cold median 8.4 s (p90 15.3 s), cache hit 0.026 s; 4 concurrent distinct requests in 12.6 s, all correct; about $0.014 per cold question | Tiny sample; one run. |

Working notes that shaped the results: a facts-first context ordering in RAGAS only moved average precision from 0.72 to 0.74 (my earlier
guess that the ordering explained the gap was wrong, though it matters a lot for individual questions); the first RAGAS run was interrupted
by an exhausted API balance and completed in a second pass; one faithfulness score failed because the judge ran out of output tokens on a
long answer, not because of credits.

## 7. Observability insights

Every request is one Langfuse trace with the LangGraph nodes as spans, token usage and cost on the generations, the chat session as the
Langfuse session, and tags and metadata for blocked / warned / follow-up / cache-hit / tool use. Screenshots (taken from the checks'
traces):

![A normal request](screenshots/langfuse_normal.png)
![A tool-using request (SQL)](screenshots/langfuse_tool_sql.png)
![A blocked injection](screenshots/langfuse_blocked.png)

What the traces were useful for:

- **Where the time goes** (cold requests, median 8.4 s): agent 3.7 s (44%), output guard 2.3 s (28%), retrieve 1.3 s (15%), input guard 1.1 s
  (14%). The two Haiku-based guards together cost about as much time as the Sonnet call, which is what makes the cache and any retry (a flagged
  first draft roughly doubles a request) matter.
- **Cost**: Sonnet is about 94% of the token cost although Haiku makes more calls.
- **The totals were wrong.** The screenshot of the SQL question shows $0.029 and 22k tokens; the real figures are about $0.0143 and 11k. Every
  model call was recorded twice (the LangChain handler's span and the Anthropic SDK's own span nested under it) and Langfuse sums both. The cost
  report had avoided it by counting one span type; the tracing client now drops the duplicate spans. Traces recorded before the change, including
  these screenshots, still show the doubled totals.
- **Tags and ingestion behave in non-obvious ways**: the UI showed only the tag stamped on the early child spans, so the platform appears to take a
  trace's tags from one span rather than merging them (an inference from behaviour, not documented). Tags are now set on the root and a final event,
  which the UI shows correctly in the screenshots. Observations are also ingested out of order, so tests wait for the count to stabilise.
- **PII masking was verified at the sink**: a request containing an email and a phone number was fetched back from Langfuse and contains
  neither (and does contain `[EMAIL]` and `[PHONE]`).

## 8. Challenges and improvements

Challenges, in the order they bit:

- **Messy data on purpose.** Contradictory dependency directions, a self-dependency and conflicting FAQ/SOP escalation paths forced a "manuals
  are authoritative, keep the conflicts" rule, and the conflict question became a benchmark for the whole system.
- **Retrieval that looks fine until it is measured.** Recall of the right documents was only 12 of 26 with vector search; the cross-encoder
  dropped answer-bearing fields; and the real fix (graph facts) was a different component from the one being tuned.
- **Evaluation tooling.** RAGAS 0.4.3 needed an import shim and a judge wrapper that discards a `temperature` these models reject; Promptfoo
  expands array variables into separate cases; several of my own tests were too strict (decline phrase lists, a refusal vs a masked answer).
  Four failures along the way were test defects, each documented as such rather than silently fixed: a stale "retrieval ran" assertion, a step-count assertion that ignored the guard's retry, a bulk-PII expectation that rejected a safe refusal, and a decline phrase list missing "couldn't find".
- **Measuring honestly.** Out-of-order Langfuse ingestion made a test pass or fail by timing; duplicate spans doubled the cost totals; a run
  that looked complete had hit the API credit limit. Each of these produced a plausible-looking number that was wrong.

Improvements, roughly by value:

1. **Follow references in retrieved chunks** (link entities named in the top chunks, pull their documents) to close the Q4 gap; consider BM25 for terms like "breach" and "logs".
2. **Tune the injection classifier** for ordinary data requests and "ignore the old version" phrasings, and grow the benign set so the false-positive rate is measured on more than 10 prompts.
3. **Give the groundedness judge the tool-call arguments**, and retry it once on a validation error (it failed open on 2 of 75 answers).
4. **A held-out adversarial set** written by someone else, and prompts that get past the classifier so the tool defences are exercised end to end; a real sandbox (container) for `python_exec`.
5. **Persistence and authentication** (sessions and cache in a store, tokens instead of a bearer session id); question-aware graph-fact selection for hub entities; a router for Haiku vs Sonnet only after it can be evaluated.
