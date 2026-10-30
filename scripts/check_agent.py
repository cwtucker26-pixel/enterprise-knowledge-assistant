"""Task 5 verification: offline tool-safety tests, the max-steps guard, and five agent questions.

For each question it prints the tool calls in order, the step count and the final answer, and flags
whether the expected tools were used. A failing LLM question is re-run once to tell a flake from a bug.
Exit code is non-zero if anything still fails.
"""
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.graph import build_agent_graph, final_answer, run_agent, tool_calls_in_order
from src.agent.tools import python_exec, sql_query
from src.config import SQLITE_PATH
from src.retrieval.hybrid import HybridRetriever


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text) if n.strip(",")]


def has_number(text: str, value: float, tol: float = 0.011) -> bool:
    return any(abs(n - value) <= tol for n in numbers_in(text))


def ground_truth() -> dict:
    with sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True) as conn:
        count, total = conn.execute(
            "SELECT COUNT(*), ROUND(SUM(amount_usd), 2) FROM transactions WHERE status = 'completed'"
        ).fetchone()
        avgs = [r[0] for r in conn.execute(
            "SELECT AVG(amount_usd) FROM transactions WHERE status = 'completed' GROUP BY product_id")]
    return {"count": count, "total": total, "stdev": statistics.pstdev(avgs)}


# --- offline tests: the tools are the risky part, so they are tested without the LLM ---
def check_tool_safety() -> list[str]:
    failures = []

    def expect(label, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(label)

    expect("sql_query runs a SELECT", "46" in sql_query("SELECT COUNT(*) FROM users"))
    expect("sql_query rejects DROP TABLE", sql_query("DROP TABLE users").startswith("Error"))
    expect("sql_query rejects UPDATE", sql_query("UPDATE users SET name = 'x'").startswith("Error"))
    expect("sql_query rejects two statements", sql_query("SELECT 1; DROP TABLE users").startswith("Error"))
    expect("sql_query rejects PRAGMA", sql_query("PRAGMA table_info(users)").startswith("Error"))
    expect("sql_query rejects ATTACH", sql_query("ATTACH DATABASE ':memory:' AS x").startswith("Error"))
    out = sql_query("SELECT * FROM transactions")
    expect("sql_query output is capped and says it was truncated", "truncated" in out and len(out) <= 4000)
    expect("sql_query reports bad SQL as text", sql_query("SELECT nope FROM users").startswith("SQL error"))
    expect("database unchanged after the attack attempts", "46" in sql_query("SELECT COUNT(*) FROM users")
           and "Priya Sharma" in sql_query("SELECT name FROM users WHERE user_id = 'U0001'"))
    t0 = time.monotonic()
    out = python_exec("while True: pass")
    expect(f"python_exec times out on an infinite loop ({time.monotonic() - t0:.0f}s)", "timed out" in out)
    out = python_exec("import os; print(sorted(os.environ))")
    secret_like = ("KEY", "SECRET", "TOKEN", "PASSWORD", "ANTHROPIC", "NEO4J", "TAVILY", "LANGFUSE")
    expect(f"python_exec environment has no secrets ({out.split('stdout:')[1].splitlines()[1]})",
           not any(w in out for w in secret_like))
    expect("python_exec returns stdout", "42" in python_exec("print(6 * 7)"))
    expect("python_exec reports errors", "ZeroDivisionError" in python_exec("1/0"))
    return failures


QUESTIONS = [
    {
        "id": "knowledge-only",
        "question": "Who is the system owner of Payment-Service, and which team owns it?",
        "tools": [],  # exact sequence expected
        "answer_has": [r"Priya Sharma", r"Billing"],
    },
    {
        "id": "sql",
        "question": "How many completed transactions are there, and what is their total amount in USD?",
        "tools": ["sql_query"],
        "numbers": "count_total",
    },
    {
        "id": "web",
        "question": "What is the latest stable release of the Python programming language?",
        "tools": ["web_search"],
    },
    {
        "id": "sql-then-python",
        "question": (
            "Using the transactions table, compute the average amount of completed transactions for each "
            "product. Then use Python to calculate the population standard deviation of those per-product "
            "averages."
        ),
        "tools": ["sql_query", "python_exec"],
        "numbers": "stdev",
    },
    {
        "id": "Q6-conflict",
        "question": "Does Auth-DB depend on Payment-Service?",
        "tools": [],
        "answer_has": [r"INC-204", r"conflict|contradict|disagree|revers|opposite|not authoritative"],
    },
]


def evaluate(q: dict, state: dict, truth: dict) -> list[str]:
    """Return a list of problems (empty = pass)."""
    problems = []
    names = [c["name"] for c in tool_calls_in_order(state)]
    # expected tools must appear in order (subsequence); extra tools are allowed except for no-tool questions
    if not q["tools"]:
        if names:
            problems.append(f"expected no tools, got {names}")
    else:
        it = iter(names)
        if not all(t in it for t in q["tools"]):
            problems.append(f"expected {q['tools']} in order, got {names}")
    answer = final_answer(state)
    if state["messages"][-1].tool_calls:
        problems.append("final message still has tool calls")
    for pattern in q.get("answer_has", []):
        if not re.search(pattern, answer, re.I):
            problems.append(f"answer does not match /{pattern}/")
    if q.get("numbers") == "count_total":
        if not has_number(answer, truth["count"], 0.5) or not has_number(answer, truth["total"]):
            problems.append(f"answer lacks {truth['count']} and/or {truth['total']}")
    if q.get("numbers") == "stdev" and not has_number(answer, truth["stdev"]):
        problems.append(f"answer lacks the population stdev {truth['stdev']:.2f}")
    return problems


def show(state: dict, seconds: float):
    calls = tool_calls_in_order(state)
    print(f"  tool calls ({len(calls)}):")
    for i, c in enumerate(calls, 1):
        args = " ".join(str(v) for v in c["args"].values()).replace("\n", " ")
        print(f"    {i}. {c['name']}: {args[:110]}")
    print(f"  steps (agent calls): {state['steps']}   time: {seconds:.1f}s")
    print("  final answer:")
    for line in final_answer(state).splitlines():
        print("    " + line)


def main() -> int:
    failures = []
    print("Tool safety (offline)")
    failures += check_tool_safety()

    truth = ground_truth()
    print(f"\nGround truth from SQLite: {truth['count']} completed, total {truth['total']}, "
          f"population stdev of per-product averages {truth['stdev']:.2f}")
    retriever = HybridRetriever()
    agent = build_agent_graph(retriever=retriever)

    for q in QUESTIONS:
        print(f"\n[{q['id']}] {q['question']}")
        for attempt in (1, 2):
            t0 = time.monotonic()
            state = run_agent(agent, q["question"])
            problems = evaluate(q, state, truth)
            print(f"  attempt {attempt}:")
            show(state, time.monotonic() - t0)
            print(f"  -> {'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}")
            if not problems:
                if attempt == 2:
                    print("  note: failed on the first attempt and passed on the retry (flaky)")
                break
        else:
            failures.append(q["id"])

    print("\n[guard] sql-then-python question with MAX_AGENT_STEPS=2: the loop must still end with an answer")
    guarded = build_agent_graph(max_steps=2, retriever=retriever)
    state = run_agent(guarded, QUESTIONS[3]["question"])
    show(state, 0)
    ok = state["steps"] == 2 and not state["messages"][-1].tool_calls and bool(final_answer(state).strip())
    print(f"  -> {'PASS' if ok else 'FAIL'}: steps={state['steps']}, final message has tool calls: "
          f"{bool(state['messages'][-1].tool_calls)}")
    if not ok:
        failures.append("guard")

    retriever.close()
    print("\n" + ("PASS" if not failures else "FAIL: " + ", ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
