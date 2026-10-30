# Test Questions (ACME Corp dataset)

Hand-written ground truth for multi-hop retrieval/agent testing. Reused by Promptfoo (Task 9)
and RAGAS (Task 10). Source of truth for `DEPENDS_ON`: the technical manuals.

## Q1 — Root cause and who to page
**Question:** Payment-Service is returning 503s. Which upstream system is the likely cause, and who do I page?

**Expected:** Auth-DB (manual_payment_service: Payment-Service depends on Auth-DB for all auth/session
validation). Page Marcus Lee (Infrastructure), who owns Auth-DB (manual_auth_db) and is named in
SOP-17 step 3. If unresolved after 20 minutes, escalate Marcus Lee -> Natalia Voss -> VP Engineering (SOP-05).
Priya Sharma (Billing) owns Payment-Service itself.

**Sources:** manual_payment_service, manual_auth_db, SOP-17, SOP-05

## Q2 — Dependents of Auth-DB and their owners
**Question:** Which systems depend on Auth-DB, and who owns each?

**Expected:** Payment-Service (Billing, Priya Sharma), UserProfile-API (Customer Success, Amara Okafor),
DataWarehouse (Data Engineering, Chen Wei), APIGateway (Security, Natalia Voss).
Note: FAQ.md lists only the first three (omits DataWarehouse); the manuals include all four.
INC-205's "Auth-DB depends on APIGateway" is a distractor.

**Sources:** the four dependent manuals, manual_auth_db, FAQ.md, users.csv (leads)

## Q3 — Auth-DB incidents, SOPs, SOP owners
**Question:** Which past incidents involved Auth-DB, which SOP resolved each, and who owns those SOPs?

**Expected:** INC-201 -> SOP-01 (Marcus Lee); INC-204 -> SOP-17 (Priya Sharma); INC-205 -> SOP-22
(Natalia Voss).

**Sources:** doc_metadata.csv, INC-201, INC-204, INC-205, SOP-01, SOP-17, SOP-22

## Q4 — PII breach response
**Question:** We suspect customer PII was exfiltrated from UserProfile-API. What do we do, who is told and by when, and can we delete the logs?

**Expected:** POL-001 says severe breaches trigger SOP-22 (Security Breach Containment). Alert Natalia
Voss within 15 minutes; isolate affected systems with Marcus Lee; revoke compromised credentials via
Auth-DB admin; do NOT delete logs (legal requirement per POL-002; system logs are retained 1 year);
notify legal/compliance because PII may be exposed.

**Sources:** POL-001, POL-002, POL-003, SOP-22

## Q5 — Remote access to Auth-DB
**Question:** I work remotely and need access to Auth-DB. What are the requirements?

**Expected:** POL-004: VPN (provisioned per SOP-03), company laptop (or MDM-approved device) with disk
encryption. SOP-03: access request ticket with manager approval, identity verified via Auth-DB MFA,
logged, access review scheduled in 90 days. POL-003: MFA mandatory for Auth-DB; quarterly access reviews.
Security (Natalia Voss) owns the process.

**Sources:** POL-004, SOP-03, POL-003

## Q6 — Deliberate conflict (hallucination / re-ranking test)
**Question:** Does Auth-DB depend on Payment-Service?

**Expected:** The best answer says no, per the manuals: Payment-Service depends on Auth-DB (not the
reverse). It should acknowledge that INC-204 states the inverse and flag the inconsistency instead of
asserting either silently. A confident "yes" based only on INC-204, or an answer that hides the
conflict, is a fail.

**Sources:** manual_payment_service, manual_auth_db, INC-204
