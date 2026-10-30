# RAGAS evaluation report

**Run:** 2026-10-30, `python eval/ragas/run_ragas.py` (RAGAS 0.4.3). 18 questions with reference answers written from the source documents
(`eval/ragas/dataset.json`; each reference's source docs are listed at the end for spot-checking). Answers come from the real API (`POST /chat`).
Judge: **claude-sonnet-5-5** with the local bge-small embeddings (answer relevancy). Raw per-question scores, answers and contexts:
`eval/results/ragas_2026-10-30.json`.

> **This run is incomplete, and one number is methodologically weak. Read "Caveats" before using these figures.** The Anthropic account ran out of
> credits during the last scoring step, so the vector-only comparison has no precision scores and recall for only 7 of 18 questions, and
> faithfulness is missing for R08. `python eval/ragas/run_ragas.py --complete` finishes exactly what is missing once credits are restored.

## How the contexts were built

The answer is produced from **retrieved chunks plus graph facts** (short lines such as `Marcus Lee RELATED_TO[system_owner] Auth-DB (source: ...)`).
Scoring faithfulness against the chunks alone would call every fact-derived claim unsupported, so the contexts passed to RAGAS are the 8 retrieved chunk
texts plus **one extra context item holding the graph facts**. The contexts were rebuilt in-process with the same retriever code (retrieval is
deterministic) and checked against the chunk ids the API reported: **18/18 questions matched**. Context precision and recall are also reported for the
chunks alone, which isolates what the facts contribute.

## Results (hybrid system, 18 questions)

| metric | score |
|---|---|
| faithfulness | 0.93 (17/18 scored) |
| answer relevancy | 0.92 |
| context precision | 0.72 |
| context recall | 0.91 |

By question category:

| category (n) | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|
| single-hop (6) | 0.91 | 0.90 | 0.85 | 0.92 |
| multi-hop (6) | 0.91 (5/6 scored; R08 missing) | 0.93 | 0.58 | 0.81 |
| conflict (3) | 0.95 | 0.97 | 0.78 | 1.00 |
| policy-procedure (3) | 0.99 | 0.86 | 0.69 | 1.00 |
| **all (18)** | 0.93 (17/18 scored) | 0.92 | 0.72 | 0.91 |

## What the graph contributes

| retrieval variant | context precision | context recall (all 18) | context recall (R01-R07 only) |
|---|---|---|---|
| hybrid: chunks + graph facts (as run: facts item last) | 0.72 | 0.91 | 0.93 |
| hybrid chunks only (no facts) | 0.80 | 0.80 | 0.93 |
| vector-only top 8 (no graph, no re-ranker) | not scored | not scored | 0.81 |

- **The facts raise recall** from 0.80 (chunks only) to 0.91: for R08 (Auth-DB dependents and owners) recall goes 0.00 to 1.00 and for R09 (incidents, SOPs, SOP owners) 0.33 to 1.00,
  because the owner and dependency relationships are in the facts, not in the chunks the re-ranker keeps. This matches the Task 4 finding.
- **The facts lower precision as measured** (0.80 to 0.72), but part of that is probably an artefact of where I put them (caveat 2); I could not confirm how much, because the facts-first re-score is pending.
- **Vector-only comparison: incomplete.** The only vector-only scores that exist are recall for R01-R07 (R01, R02, R03, R04, R05, R06, R07): 0.81, against 0.93 for hybrid chunks on the same
  7 questions. Six of those seven are easy single-hop questions, so this says almost nothing about the hybrid's benefit on the multi-hop ones; no conclusion should be drawn from it.

## Weakest questions and why

| id | symptom | cause (checked against the retrieved chunk ids) |
|---|---|---|
| **R10** (PII exfiltration, who is told, delete logs?) | recall 0.17 (both with and without facts) | **Retrieval miss.** SOP-22 and POL-003 are not retrieved at all (only POL-001#2, which merely names SOP-22). The deadline and the "preserve the logs" rule are in SOP-22/POL-003. Same gap as Promptfoo MH-04. Fix: follow entities named in retrieved chunks. |
| **R09** (Auth-DB incidents, SOPs, SOP owners) | precision 0.11 hybrid, 0.00 chunks only | The retrieved chunks are mostly incident text (Resolution/Overview), while the question and reference are about SOP *owners*; no SOP document is retrieved. The correct answer comes from the facts (recall 1.00 with them, 0.33 without). |
| **R12** (lead of the team owning UserProfile-API's dependency) | precision 0.39 | `manual_auth_db` is not among the retrieved documents; the answer again relies on the graph facts (recall 1.00). |
| **R07** (503s on Payment-Service) | precision 0.38 hybrid vs 0.88 chunks only | Mostly the ordering artefact (facts item scored at position 9, see Caveats), plus two INC-202 chunks that are not relevant to the reference. |
| **R06** (what happened in INC-205) | recall 0.50 | **Retrieval miss.** Only the Summary and Resolution chunks of INC-205 were retrieved, not its header chunk, where the incident lead (Natalia Voss) is listed. |
| **R04, R16** (single policy facts) | precision 0.50 | The right policy chunk is retrieved, but with unrelated chunks around it (8 chunks are always returned, even when one suffices). |
| **R03** | faithfulness 0.67 | Likely **judge noise**: the answer is two short statements, both stated in the cited chunk `POL-002#1` ("System logs: 1 year. Backup snapshots: 30 days rolling"). I could not see which statement the judge marked unsupported. |

Faithfulness is 0.87 to 1.00 on the 17 scored questions except R03 (0.67) and R07 (0.81); answer relevancy is 0.84 to 1.00; recall is 1.00 for 15 of 18 questions.

## Caveats

1. **Incomplete run (credits).** Missing: vector-only precision (0/18) and recall (11/18), and faithfulness for R08. The hybrid and chunks-only variants are fully scored for precision, recall and answer relevancy. The Promptfoo and safety runs finished before the credits ran out and contain no credit errors (checked).
2. **Context precision depends on order, and the facts item went last.** RAGAS' reference-based precision is a rank-weighted average over the context list. In the real prompt the graph facts come *before* the passages; here the facts item is position 9, where a relevant item scores a low precision@k and drags the average down (R07: 0.88 without the facts item, 0.38 with it). So the hybrid precision of 0.72 may understate the system; by how much is unmeasured. `--complete` also scores a `hybrid_facts_first` variant that mirrors the prompt; I have not seen its numbers.
3. **The judge is from the same model family as the generator** (Sonnet judging Sonnet answers), which can favour the generator's phrasing. A different family would be a stronger judge.
4. **Small sample, self-written references.** 18 questions; the references were written by me from the documents (sources listed below), so they share my reading of them. Scores near 1.0 on 18 questions do not generalise; differences of a few hundredths are noise. I did not repeat the run, so run-to-run judge variance is unmeasured.
5. **RAGAS itself:** version 0.4.3 needed a small compatibility shim (`eval/ragas/compat.py`) to import with the installed `langchain-community`, and a judge wrapper that discards the `temperature` it tries to set (`eval/ragas/judge.py`), because these Anthropic models reject it.

## Per-question scores

| id | category | faithfulness | relevancy | precision | recall | chunks-only precision | chunks-only recall |
|---|---|---|---|---|---|---|---|
| R01 | single-hop | 1.00 | 0.89 | 0.91 | 1.00 | 1.00 | 1.00 |
| R02 | single-hop | 1.00 | 0.86 | 1.00 | 1.00 | 1.00 | 1.00 |
| R03 | single-hop | 0.67 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| R04 | single-hop | 0.90 | 0.93 | 0.50 | 1.00 | 0.50 | 1.00 |
| R05 | single-hop | 0.91 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 |
| R06 | single-hop | 1.00 | 0.91 | 0.70 | 0.50 | 0.70 | 0.50 |
| R07 | multi-hop | 0.81 | 0.84 | 0.38 | 1.00 | 0.88 | 1.00 |
| R08 | multi-hop | n/a | 0.94 | 0.61 | 1.00 | 1.00 | 0.00 |
| R09 | multi-hop | 0.89 | 0.95 | 0.11 | 1.00 | 0.00 | 0.33 |
| R10 | multi-hop | 0.87 | 0.93 | 1.00 | 0.17 | 1.00 | 0.17 |
| R11 | multi-hop | 0.95 | 0.94 | 1.00 | 0.67 | 1.00 | 0.67 |
| R12 | multi-hop | 1.00 | 0.99 | 0.39 | 1.00 | 0.42 | 1.00 |
| R13 | conflict | 0.94 | 1.00 | 0.78 | 1.00 | 0.88 | 0.67 |
| R14 | conflict | 1.00 | 0.94 | 0.58 | 1.00 | 0.79 | 1.00 |
| R15 | conflict | 0.91 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| R16 | policy-procedure | 1.00 | 0.87 | 0.50 | 1.00 | 0.50 | 1.00 |
| R17 | policy-procedure | 1.00 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 |
| R18 | policy-procedure | 0.96 | 0.86 | 0.58 | 1.00 | 0.75 | 1.00 |

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
