// Promptfoo provider that calls the real API (POST /chat). The base URL comes from CHAT_API_URL
// (scripts/run_promptfoo.py starts the server and sets it).
// Multi-turn: if a test has `vars.turns_json` (a JSON string holding a list of questions), the turns are sent in order
// in ONE session and the last response is the output, so follow-up cases are self-contained and safe to run in
// parallel. (It is a string because Promptfoo expands array-valued vars into separate test cases.)
module.exports = class ChatProvider {
  constructor(options) {
    this.providerId = (options && options.id) || 'acme-chat-api';
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt, context) {
    const base = process.env.CHAT_API_URL || 'http://127.0.0.1:8765';
    const vars = (context && context.vars) || {};
    const parsed = vars.turns_json ? JSON.parse(vars.turns_json) : null;
    const turns = Array.isArray(parsed) && parsed.length ? parsed : [prompt];
    let sessionId;
    let last = {};
    const transcript = [];
    for (const question of turns) {
      const res = await fetch(`${base}/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(sessionId ? { question, session_id: sessionId } : { question }),
      });
      const text = await res.text();
      let body = {};
      try {
        body = JSON.parse(text);
      } catch (e) {
        body = { detail: 'non-JSON response' };
      }
      last = { status: res.status, ...body };
      sessionId = body.session_id || sessionId;
      transcript.push({ question, status: res.status, answer: body.answer, rewritten_query: body.rewritten_query });
    }
    return { output: { ...last, transcript } };
  }
};
