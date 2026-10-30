// Turns every entry of eval/safety/adversarial_prompts.json into a Promptfoo test, so the safety dataset is shared
// between this suite and the safety scorecard (scripts/run_safety.py).
const path = require('path');
const prompts = require(path.resolve(__dirname, '../safety/adversarial_prompts.json')).prompts;

module.exports = () =>
  prompts.map((c) => ({
    description: `ADV-${c.id} [${c.category}] expect ${c.expected}`,
    vars: { question: c.prompt },
    assert: [{ type: 'javascript', value: `file://${__dirname}/assertions.js:adversarial`, config: { case: c } }],
  }));
