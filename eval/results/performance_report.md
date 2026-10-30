# Performance report (2026-10-30)

Measured with `python scripts/perf_test.py` against the real API (`POST /chat`), deliberately small request counts to keep API cost low. **With so few requests, treat medians and p90 as indicative, not as a benchmark**; p90 over 5 requests is essentially the slowest one.

## Latency: cold vs cached

| | n | median | p90 | min | max |
|---|---|---|---|---|---|
| cold (full pipeline) | 5 | 8.4 s | 15.3 s | 5.7 s | 15.3 s |
| cached (semantic-cache hit) | 5 | 0.026 s | 0.034 s | 0.025 s | 0.034 s |

A cache hit is about **323x faster** than a cold answer at the median (a hit costs one local embedding call and no LLM call).

Cold requests by question type:

| type | question | seconds | tools | correct |
|---|---|---|---|---|
| single-hop | Who owns Payment-Service? | 8.4 | - | True |
| single-hop | What is the data retention period for transaction records? | 5.7 | - | True |
| multi-hop | Which systems depend on Auth-DB, and who owns each? | 15.3 | - | True |
| conflict | Does Auth-DB depend on Payment-Service? | 12.0 | - | True |
| sql | How many completed transactions are there, and what is their total amo | 6.6 | sql_query | True |

Repeating the same 5 questions in new sessions gave 5 cache hits (5 were expected: answers that were warned, blocked or used web search are never cached); the replayed answer matched the original in 5/5 hits.

## Cache correctness probes

| probe | expected | got | pass |
|---|---|---|---|
| lower-case, no punctuation variant: `who owns payment service` | HIT | HIT | yes |
| no question mark: `Who owns Payment-Service` | HIT | HIT | yes |
| REVERSED dependency direction (cosine ~0.996 to a cached question): `Does Payment-Service depend on Auth-DB?` | MISS | MISS | yes |
| cache bypassed with use_cache=false: `Who owns Payment-Service?` | BYPASS | BYPASS | yes |
| first turn answered from the cache: `Who owns Payment-Service?` | HIT | HIT | yes |
| follow-up after a cache hit is not cached and still works: `What does it depend on?` | SKIP | SKIP | yes |

The reversed-direction question is the important one: its embedding is almost identical to the cached question (cosine ~0.996, higher than genuine paraphrases), so an embedding-similarity threshold alone would have replayed the wrong answer. The cache therefore also requires the same structural signature (ordered entities plus direction/negation/number words); see `src/api/cache.py` and `scripts/check_cache.py`.

## Throughput (concurrent requests)

- **4 distinct questions at once** (cache bypassed): wall time 12.6 s, per-request latencies [8.3, 10.3, 6.6, 12.6] s, all HTTP 200 and each answer contained its expected facts: **True** (no cross-talk between requests). Throughput 19.1 requests/min. For comparison, answering the same 4 one after another would take roughly 4 x the cold median = 33.7 s (an estimate, not measured).

This exercises the shared embedding model, re-ranker, Neo4j driver and checkpointer under concurrency; no errors or wrong answers appeared. It is a 4-request test: it shows the shared models are usable from several threads, not where the system saturates (Anthropic rate limits and CPU inference are the likely limits).

## Where the time goes (cold requests, from Langfuse spans)

| node | median seconds per request | share of median latency |
|---|---|---|
| input_guard | 1.1 | 14% |
| contextualize | 0.0 | 0% |
| retrieve | 1.3 | 15% |
| agent | 3.7 | 44% |
| tools | 0.0 | 0% |
| output_guard | 2.3 | 28% |

(`agent` is the Sonnet call(s) including its reasoning; `input_guard` and `output_guard` contain the Haiku classifier and groundedness judge calls; `retrieve` is embedding, Chroma, Neo4j and the cross-encoder. Nodes can run more than once per request, so the shares do not need to sum to 100%.)

## Cost per question (from Langfuse token usage)

| model | calls | input tokens | output tokens (incl. reasoning) | cost |
|---|---|---|---|---|
| claude-haiku-5-5 | 10 | 25,562 | 3,637 | $0.0044 |
| claude-sonnet-5-5 | 6 | 20,648 | 2,319 | $0.0645 |

Per cold question: median **$0.0143**, range $0.0081 to $0.0189 (mean $0.0138). A cache hit costs no model tokens. Computed from the `ChatAnthropic` generation spans (the Anthropic-SDK spans record each call a second time and are excluded), at USD per million tokens: Sonnet 5.5 $2 in / $10 out, Haiku 5.5 $0.10 in / $0.50 out (list prices in `PRICES` in `scripts/perf_test.py`; Haiku cache-read price assumed). Prompt-cache usage was not measured.

**Observability finding: Langfuse totals were double the real figures, now fixed.** While reading the traces I noticed the Langfuse UI showed about $0.029 and 22k tokens for the SQL question, against ~$0.0143 and ~11k tokens computed here. Every model call was recorded twice (the LangChain handler's `ChatAnthropic` generation and the Anthropic SDK's own `anthropic.messages.create` span, nested under it) and Langfuse sums both. The tracing client now drops the SDK-level spans (`export_span` in `src/observability/tracing.py`); a trace taken after the change has only `ChatAnthropic` generations and totals of about 10.5k input + 0.4k output tokens. The figures in this report were computed from the `ChatAnthropic` spans alone, so they were never affected; traces recorded before the change (including the screenshots in `docs/screenshots/`) still show the doubled totals.

## Regression check with the cache enabled

The full Promptfoo suite (75 cases) was re-run after the cache was added (`promptfoo_2026-10-30_cache.json`): **72/75 passed**, failures MH-04, SQP-02, ADV-BN-09, and **no regression against the baseline** (the same cases that failed in both earlier runs). None of the 75 responses were cache hits: the suite's first-turn questions are almost all distinct, so it exercises the cache's miss-and-store path and shows it did not break any behaviour, but it does not exercise replay. Hits are covered by `scripts/check_api.py` and the probes above.

## Model routing: not implemented, on purpose

Routing 'simple' questions to Haiku was considered and skipped:

1. **The failure mode is silent.** A heuristic router (question length, one linked entity) will sometimes send a multi-hop question to the cheaper model, and the result is a plausible but wrong answer, not an error. The hardest questions here (Q2/Q3, the dependency conflict) look short and simple.
2. **Adding it responsibly needs its own evaluation** (router accuracy, plus the Promptfoo/RAGAS suites on both paths), which is more cost and risk than the saving justifies at this scale.
3. **The saving would be real, but small in absolute terms.** Sonnet is about 94% of the measured cost (table above) and Haiku is roughly 20x cheaper per token, so routing a question to Haiku could cut that question's cost by most of its agent portion. But a cold question costs about 1.4 cents, and the semantic cache already removes the whole cost of repeated questions. At this scale the quality risk outweighes a saving of a fraction of a cent per question.
4. A safer first step, if cost ever matters: lower the agent's reasoning effort for first-pass answers and measure quality with the existing suites.

## Limitations

- 5 cold, 5 cached, 5 probe and 4 concurrent requests: small samples; a single run; no warm-up control for the embedding models.
- The cache lives in process memory (lost on restart, one instance only), is keyed on the *masked* question, and hits are limited by design to near-identical wording.
- Latency includes the model-based guards; a retry (a flagged first draft) roughly doubles a request.
- Cost uses list prices and the token counts Langfuse recorded; thinking tokens are billed as output.
