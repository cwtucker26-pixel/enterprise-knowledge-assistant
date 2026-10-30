// Hand-written regression cases. They assert facts, fields and behaviour on the JSON response of POST /chat,
// never exact wording. Q1..Q6 mirror eval/test_questions.md. Adversarial/PII/entity-confusion/benign cases are
// generated from eval/safety/adversarial_prompts.json (adversarial_tests.js).
const A = `file://${__dirname}/assertions.js`;

const notBlocked = { type: 'javascript', value: `${A}:notBlocked` };
const cited = { type: 'javascript', value: `${A}:hasCitations` };
const noWarning = { type: 'javascript', value: `${A}:noWarning` };
const has = (...facts) => ({ type: 'javascript', value: `${A}:answerHas`, config: { facts } });
const hasAny = (...facts) => ({ type: 'javascript', value: `${A}:answerHasAny`, config: { facts } });
const lacks = (...facts) => ({ type: 'javascript', value: `${A}:answerLacks`, config: { facts } });
const rewritten = (...facts) => ({ type: 'javascript', value: `${A}:rewrittenHas`, config: { facts } });
const tools = (...names) => ({ type: 'javascript', value: `${A}:usedTools`, config: { tools: names } });
const noTools = { type: 'javascript', value: `${A}:usedTools`, config: { none: true } };
const rubric = (text) => ({ type: 'llm-rubric', value: text });

const test = (description, question, assert, extraVars = {}) => ({
  description,
  vars: { question, ...extraVars },
  assert,
});

module.exports = () => [
  // --- single-hop facts ---
  test('SH-01 single-hop: owner of Payment-Service', 'Who owns Payment-Service?', [notBlocked, cited, noWarning, noTools, has('Billing', 'Priya Sharma')]),
  test('SH-02 single-hop: retention of transaction records', 'What is the data retention period for transaction records?', [notBlocked, cited, has('7 years|seven years')]),
  test('SH-03 single-hop: procedure owner of SOP-22', 'Who is the procedure owner of SOP-22?', [notBlocked, cited, has('Natalia Voss')]),
  test('SH-04 single-hop: personal devices policy', 'What does the remote work policy (POL-004) require for personal devices?', [notBlocked, cited, has('MDM')]),
  test('SH-05 single-hop: incident reporting deadline', 'Within how many minutes must security incidents be reported to Natalia Voss?', [notBlocked, cited, has('15')]),
  test('SH-06 single-hop: escalation path in SOP-05', 'What is the team lead escalation path in SOP-05?', [notBlocked, cited, has('Marcus Lee', 'Natalia Voss', 'VP')]),

  // --- multi-hop (graph) ---
  test('MH-01 multi-hop (Q1): 503s on Payment-Service', 'Payment-Service is returning 503s. Which upstream system is the likely cause, and who do I page?', [notBlocked, cited, has('Auth-DB', 'Marcus Lee')]),
  test('MH-02 multi-hop (Q2): dependents of Auth-DB and owners', 'Which systems depend on Auth-DB, and who owns each?', [notBlocked, cited, has('APIGateway', 'DataWarehouse', 'Payment-Service', 'UserProfile-API', 'Natalia Voss', 'Chen Wei', 'Priya Sharma', 'Amara Okafor')]),
  test('MH-03 multi-hop (Q3): Auth-DB incidents, SOPs, SOP owners', 'Which past incidents involved Auth-DB, which SOP resolved each, and who owns those SOPs?', [notBlocked, cited, has('INC-201', 'INC-204', 'INC-205', 'SOP-01', 'SOP-17', 'SOP-22', 'Marcus Lee', 'Priya Sharma', 'Natalia Voss')]),
  test('MH-04 multi-hop (Q4): PII exfiltration response', 'We suspect customer PII was exfiltrated from UserProfile-API. What do we do, who is told and by when, and can we delete the logs?', [notBlocked, cited, has('SOP-22', 'Natalia Voss', '15 minutes|15 min'), hasAny('do not delete|don\'t delete|not delete|preserve|must not be deleted|cannot delete|should not delete')]),
  test('MH-05 multi-hop (Q5): remote access to Auth-DB', 'I work remotely and need access to Auth-DB. What are the requirements?', [notBlocked, cited, has('VPN', 'MFA|multi-factor', 'manager')]),
  test('MH-06 multi-hop: lead of the team owning UserProfile-API\'s dependency', 'Who leads the team that owns the system UserProfile-API depends on?', [notBlocked, cited, has('Auth-DB', 'Infrastructure', 'Marcus Lee')]),
  test('MH-07 multi-hop: owner of the SOP that resolved INC-205', 'Which team owns the SOP used to resolve INC-205, and who is its procedure owner?', [notBlocked, cited, has('SOP-22', 'Security', 'Natalia Voss')]),

  // --- the Auth-DB dependency conflict ---
  test('CF-01 conflict (Q6): does Auth-DB depend on Payment-Service?', 'Does Auth-DB depend on Payment-Service?', [
    notBlocked, cited, noWarning, has('INC-204'),
    hasAny('conflict', 'contradict', 'disagree', 'revers', 'opposite', 'not authoritative'),
    rubric('The answer must conclude that the technical manuals show Payment-Service depends on Auth-DB (not the reverse), and must flag that incident report INC-204 states the opposite as a conflicting, non-authoritative source. It must NOT assert as fact that Auth-DB depends on Payment-Service.'),
  ]),
  test('CF-02 conflict: APIGateway vs Auth-DB', 'Does APIGateway depend on Auth-DB, or does Auth-DB depend on APIGateway?', [
    notBlocked, cited, has('INC-205'),
    rubric('The answer must say that APIGateway depends on Auth-DB according to the technical manual, and flag that incident report INC-205 claims the reverse as conflicting. It must not present the reverse direction as established fact.'),
  ]),
  test('CF-03 conflict: which reports contradict the manuals', 'Which incident reports contradict the technical manuals about the direction of system dependencies?', [notBlocked, cited, has('INC-204', 'INC-205')]),

  // --- SQL ---
  test('SQL-01 sql: completed transactions', 'How many completed transactions are there, and what is their total amount in USD?', [notBlocked, tools('sql_query'), has('152', '148409.38')]),
  test('SQL-02 sql: Billing headcount', 'How many employees are on the Billing team?', [notBlocked, tools('sql_query'), has('9')]),
  test('SQL-03 sql: inactive employees', 'How many employees are inactive?', [notBlocked, tools('sql_query'), has('5')]),
  test('SQL-04 sql: failed transactions', 'How many transactions have failed, and what is their total amount in USD?', [notBlocked, tools('sql_query'), has('18', '19923.99')]),

  // --- SQL then Python ---
  test('SQP-01 sql->python: stdev of per-product averages', 'Using the transactions table, compute the average amount of completed transactions for each product. Then use Python to calculate the population standard deviation of those per-product averages.', [notBlocked, tools('sql_query', 'python_exec'), has('152.16')]),
  test('SQP-02 sql->python: pending share of total value', 'Get the total amount of pending transactions and the total amount of all transactions from the database, then use Python to compute the pending share as a percentage.', [notBlocked, tools('sql_query', 'python_exec'), has('16.13|16.1')]),

  // --- web ---
  test('WEB-01 web: latest Python release', 'What is the latest stable release of the Python programming language?', [notBlocked, tools('web_search'), has('python')]),
  test('WEB-02 web: latest PostgreSQL release', 'What is the latest stable major version of PostgreSQL?', [notBlocked, tools('web_search'), has('postgres')]),

  // --- follow-ups (multi-turn, same session) ---
  test('FU-01 follow-up: pronoun refers to Payment-Service', 'What does it depend on?', [notBlocked, rewritten('Payment-Service'), has('Auth-DB'), noWarning], { turns_json: JSON.stringify(['Who owns Payment-Service?', 'What does it depend on?']) }),
  test('FU-02 follow-up: "she" refers to Natalia Voss', 'Which SOPs does she own?', [notBlocked, rewritten('Natalia Voss'), has('SOP-03', 'SOP-22')], { turns_json: JSON.stringify(['Who is Natalia Voss?', 'Which SOPs does she own?']) }),
];
