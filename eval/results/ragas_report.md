# RAGAS evaluation report

**Run:** 2026-10-30, `python eval/ragas/run_ragas.py` then `--complete` (RAGAS 0.4.3). 18 questions with reference answers written from the
source documents (`eval/ragas/dataset.json`; each reference's source docs are listed at the end for spot-checking). Answers come from the real
API (`POST /chat`). Judge: **claude-sonnet-5-5** with the local bge-small embeddings (answer relevancy). Raw per-question scores, answers and
contexts: `eval/results/ragas_2026-10-30.json`. **All metrics are scored for all 18 questions in all four retrieval variants.**

## How the contexts were built

The answer is produced from **retrieved chunks plus graph facts** (short lines such as `Marcus Lee RELATED_TO[system_owner] Auth-DB (source: ...)`).
Scoring faithfulness against the chunks alone would call every fact-derived claim unsupported, so the contexts passed to RAGAS are the 8 retrieved
chunk texts plus **one extra context item holding the graph facts**. The contexts were rebuilt in-process with the same retriever code (retrieval is
deterministic) and checked against the chunk ids the API reported: **18/18 questions matched**. Context precision and recall are also computed for
three other variants to isolate what each part contributes (table below).

## Results (hybrid system, 18 questions)

| metric | score |
|---|---|
| faithfulness | 0.92 |
| answer relevancy | 0.92 |
| context precision | 0.72 (facts item last) / 0.74 (facts item first) |
| context recall | 0.91 (1.00 on 15 of 18 questions) |

By question category (context precision with the facts item last, as first run):

| category (n) | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|
| single-hop (6) | 0.91 | 0.90 | 0.85 | 0.92 |
| multi-hop (6) | 0.89 | 0.93 | 0.58 | 0.81 |
| conflict (3) | 0.95 | 0.97 | 0.78 | 1.00 |
| policy-procedure (3) | 0.99 | 0.86 | 0.69 | 1.00 |
| **all (18)** | 0.92 | 0.92 | 0.72 | 0.91 |

## What the graph and the re-ranker contribute

| retrieval variant | precision | recall | recall: multi-hop (6) | recall: conflict (3) |
|---|---|---|---|---|
| hybrid: chunks + graph facts, facts item last (as first run) | 0.72 | 0.91 | 0.81 | 1.00 |
| hybrid: chunks + graph facts, facts item first (as in the prompt) | 0.74 | 0.91 | 0.81 | 1.00 |
| hybrid chunks only (no graph facts) | 0.80 | 0.80 | 0.53 | 0.89 |
| vector-only top 8 (no graph, no re-ranker) | 0.76 | 0.69 | 0.47 | 0.56 |

- **Recall is where the hybrid wins.** Recall goes 0.69 (vector-only) to 0.80 (graph-selected documents + cross-encoder, chunks only) to 0.91 (plus the graph facts).
  The effect concentrates on the questions that need several documents: multi-hop recall 0.47 to 0.53 to 0.81, conflict 0.56 to 0.89 to 1.00.
  Single-hop and policy questions gain little (they were already answerable from one chunk).
- **The facts carry the relationships.** For R08 (Auth-DB dependents and owners) recall is 0.00 for both chunk-only variants and 1.00 with the facts; for R09 (incidents, SOPs, SOP owners) 0.33 and 1.00. The owner and dependency relationships are in the facts, not in the chunks the re-ranker keeps. This matches the Task 4 finding.
- **Precision does not improve; it stays about level.** Vector-only 0.76, chunks-only 0.80, hybrid 0.72 to 0.74. The hybrid retrieves more of what is needed without a measurable loss of precision, but it does not retrieve *less noise*: it always returns 8 chunks.
- **Precision is rank-sensitive and noisy here.** Moving the facts item from last to first changes single questions a lot (R07 0.38 to 0.84, R08 0.61 to 1.00, R12 0.39 to 0.70, but R10 1.00 to 0.33, R11 1.00 to 0.64, R06 0.70 to 0.42) while the average barely moves (0.72 to 0.74). Earlier in this work I suggested the lower hybrid precision was mostly an artefact of putting the facts last; the facts-first run does not support that: ordering explains some questions, not the average. Differences of about 0.1 or less in precision on 18 questions should not be read as real.

## Weakest questions and why

| id | symptom | cause (checked against the retrieved chunk ids) |
|---|---|---|
| **R10** (PII exfiltration, who is told, delete logs?) | recall 0.17 in every variant | **Retrieval miss.** SOP-22 and POL-003 are never retrieved (only POL-001#2, which merely names SOP-22). The deadline and the "preserve the logs" rule are in SOP-22/POL-003. Same gap as Promptfoo MH-04. Fix: follow entities named in retrieved chunks. |
| **R09** (Auth-DB incidents, SOPs, SOP owners) | precision 0.00 to 0.11 in every variant | The retrieved chunks are incident text while the reference is about SOP *owners*; no SOP document is retrieved. The right answer comes from the facts (recall 1.00 with them, 0.33 without). |
| **R12** (lead of the team owning UserProfile-API's dependency) | precision 0.39 | `manual_auth_db` is not among the retrieved documents; the answer relies on the graph facts (recall 1.00). |
| **R07** (503s on Payment-Service) | precision 0.38 (facts last) vs 0.84 (facts first) | Rank sensitivity (the facts item is relevant but sat at position 9), plus two INC-202 chunks that are not relevant to the reference. |
| **R06** (what happened in INC-205) | recall 0.50 (vector-only 0.00) | **Retrieval miss.** Only the Summary and Resolution chunks of INC-205 were retrieved, not its header chunk, where the incident lead (Natalia Voss) is listed. |
| **R04, R16** (single policy facts) | precision 0.50 | The right policy chunk is retrieved, but with unrelated chunks around it (8 are always returned). Vector-only scores R16 at 1.00, suggesting the graph-selected extra documents add noise here. |
| **R03** | faithfulness 0.67 | Likely **judge noise**: the answer is two short statements, both stated in the cited chunk `POL-002#1`. I could not see which statement the judge marked unsupported. |

Faithfulness is 0.81 to 1.00 except R03 (0.67); answer relevancy is 0.84 to 1.00; recall is 1.00 for 15 of 18 questions.

## Caveats

1. **The first run was interrupted, then completed.** The Anthropic account ran out of credits during the last scoring step of the first run (vector-only variant). The remaining scores (vector-only, facts-first, and R08's faithfulness) came from a second pass (`--complete`) with the same judge. R08's faithfulness had failed in both passes for a different reason: the judge ran out of output tokens (`LLMDidNotFinishException`) on that long answer, so I raised the judge's `max_tokens` from 4,096 to 12,000. Scores from the two passes use the same prompts and metrics but are separate judge calls.
2. **The judge is from the same model family as the generator** (Sonnet judging Sonnet answers), which can favour the generator's phrasing. A different family would be a stronger judge.
3. **Small sample, self-written references.** 18 questions; the references were written by me from the documents (sources listed below), so they share my reading of them. Scores near 1.0 on 18 questions do not generalise, and I did not repeat the run, so run-to-run judge variance is unmeasured.
4. **Vector-only here means the raw top-8 vector search** (the Task 2 baseline), without the re-ranker or the graph; "chunks only" is the hybrid pipeline's chunks without the facts item. The answers were generated only for the hybrid system, so faithfulness and answer relevancy exist only for it.
5. **RAGAS itself:** version 0.4.3 needed a small compatibility shim (`eval/ragas/compat.py`) to import with the installed `langchain-community`, and a judge wrapper that discards the `temperature` it tries to set (`eval/ragas/judge.py`), because these Anthropic models reject it.

## Per-question scores (P = context precision, R = context recall)

| id | category | faith. | relev. | hybrid P / R | facts-first P | chunks-only P / R | vector-only P / R |
|---|---|---|---|---|---|---|---|
| R01 | single-hop | 1.00 | 0.89 | 0.91 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| R02 | single-hop | 1.00 | 0.86 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| R03 | single-hop | 0.67 | 0.98 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| R04 | single-hop | 0.90 | 0.93 | 0.50 / 1.00 | 0.50 | 0.50 / 1.00 | 0.33 / 1.00 |
| R05 | single-hop | 0.91 | 0.84 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| R06 | single-hop | 1.00 | 0.91 | 0.70 / 0.50 | 0.42 | 0.70 / 0.50 | 1.00 / 0.00 |
| R07 | multi-hop | 0.81 | 0.84 | 0.38 / 1.00 | 0.84 | 0.88 / 1.00 | 1.00 / 0.67 |
| R08 | multi-hop | 0.81 | 0.94 | 0.61 / 1.00 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| R09 | multi-hop | 0.89 | 0.95 | 0.11 / 1.00 | 0.00 | 0.00 / 0.33 | 0.00 / 0.33 |
| R10 | multi-hop | 0.87 | 0.93 | 1.00 / 0.17 | 0.33 | 1.00 / 0.17 | 0.17 / 0.17 |
| R11 | multi-hop | 0.95 | 0.94 | 1.00 / 0.67 | 0.64 | 1.00 / 0.67 | 0.59 / 0.67 |
| R12 | multi-hop | 1.00 | 0.99 | 0.39 / 1.00 | 0.70 | 0.42 / 1.00 | 0.37 / 1.00 |
| R13 | conflict | 0.94 | 1.00 | 0.78 / 1.00 | 0.91 | 0.88 / 0.67 | 0.90 / 0.67 |
| R14 | conflict | 1.00 | 0.94 | 0.58 / 1.00 | 0.75 | 0.79 / 1.00 | 0.62 / 0.00 |
| R15 | conflict | 0.91 | 0.99 | 1.00 / 1.00 | 0.89 | 1.00 / 1.00 | 0.95 / 1.00 |
| R16 | policy-procedure | 1.00 | 0.87 | 0.50 / 1.00 | 0.50 | 0.50 / 1.00 | 1.00 / 1.00 |
| R17 | policy-procedure | 1.00 | 0.84 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| R18 | policy-procedure | 0.96 | 0.86 | 0.58 / 1.00 | 0.87 | 0.75 / 1.00 | 0.78 / 1.00 |

## Reference sources (for spot-checking)

- **R01** (single-hop): FAQ.md (Q: Who owns the Payment-Service?); markdown/manual_payment_service.md
- **R02** (single-hop): pdfs/POL-002.pdf (2. Retention Schedules)
- **R03** (single-hop): pdfs/POL-002.pdf (2. Retention Schedules)
- **R04** (single-hop): pdfs/POL-004.pdf (3. Equipment)
- **R05** (single-hop): markdown/SOP-05.md (Procedure)
- **R06** (single-hop): markdown/INC-205.md
- **R07** (multi-hop): markdown/manual_payment_service.md; markdown/manual_auth_db.md; markdown/SOP-17.md
- **R08** (multi-hop): markdown/manual_payment_service.md; markdown/manual_userprofile_api.md; markdown/manual_datawarehouse.md; markdown/manual_apigateway.md; markdown/manual_auth_db.md
- **R09** (multi-hop): doc_metadata.csv; markdown/INC-201.md; markdown/INC-204.md; markdown/INC-205.md; markdown/SOP-01.md; markdown/SOP-17.md; markdown/SOP-22.md
- **R10** (multi-hop): markdown/SOP-22.md; pdfs/POL-001.pdf; pdfs/POL-002.pdf; pdfs/POL-003.pdf
- **R11** (multi-hop): pdfs/POL-004.pdf; markdown/SOP-03.md; pdfs/POL-003.pdf
- **R12** (multi-hop): markdown/manual_userprofile_api.md; markdown/manual_auth_db.md
- **R13** (conflict): markdown/manual_payment_service.md; markdown/manual_auth_db.md; markdown/INC-204.md
- **R14** (conflict): markdown/INC-204.md; markdown/INC-205.md; markdown/manual_payment_service.md; markdown/manual_apigateway.md
- **R15** (conflict): markdown/SOP-05.md; FAQ.md (Q: Who is the on-call lead for Infrastructure incidents?)
- **R16** (policy-procedure): pdfs/POL-003.pdf (4. Encryption)
- **R17** (policy-procedure): pdfs/POL-001.pdf (4. Prohibited Activities)
- **R18** (policy-procedure): markdown/SOP-17.md (Procedure)
