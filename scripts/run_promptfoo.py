"""Run the Promptfoo regression suite against the real API: start the server, run promptfoo, stop the server.

    python scripts/run_promptfoo.py                  # run everything, compare with the saved baseline
    python scripts/run_promptfoo.py --filter 'SH-'   # only cases whose description matches
    python scripts/run_promptfoo.py --update-baseline  # accept this run as the new baseline

A "regression" is a case that passed in eval/promptfoo/baseline.json and fails now; the exit code is 1 if any.
Raw results go to eval/results/promptfoo_<date>.json (provider cache disabled so every run is a real run).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config  # noqa: F401,E402  (loads .env into the environment: ANTHROPIC_API_KEY for the grader)
from scripts.eval_server import serve  # noqa: E402

CONFIG = ROOT / "eval" / "promptfoo" / "promptfooconfig.yaml"
BASELINE = ROOT / "eval" / "promptfoo" / "baseline.json"


def run_promptfoo(base_url: str, out_path: Path, filter_pattern: str | None, concurrency: int) -> int:
    cmd = ["npx", "promptfoo", "eval", "-c", str(CONFIG), "-o", str(out_path), "--no-cache", "--no-progress-bar",
           "--max-concurrency", str(concurrency)]
    if filter_pattern:
        cmd += ["--filter-pattern", filter_pattern]
    env = {**os.environ, "CHAT_API_URL": base_url, "PROMPTFOO_DISABLE_TELEMETRY": "1", "PROMPTFOO_DISABLE_UPDATE": "1"}
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]["results"]
    return [{"description": r["testCase"].get("description") or r.get("description"), "success": bool(r["success"]),
             "reason": (r.get("gradingResult") or {}).get("reason", "")} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default=None, help="regex on the case description")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    out_path = ROOT / "eval" / "results" / f"promptfoo_{datetime.date.today().isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with serve() as base_url:
        code = run_promptfoo(base_url, out_path, args.filter, args.concurrency)
    if not out_path.exists():
        print("promptfoo did not produce results; exit code", code)
        return 1

    rows = load_results(out_path)
    passed = [r for r in rows if r["success"]]
    failed = [r for r in rows if not r["success"]]
    print(f"\n{len(passed)}/{len(rows)} cases passed ({len(failed)} failed). Results: {out_path.relative_to(ROOT)}")
    for r in failed:
        print(f"  FAIL  {r['description']}\n        {r['reason'][:240]}")

    if args.update_baseline:
        BASELINE.write_text(json.dumps({"date": datetime.date.today().isoformat(), "passed": sorted(r["description"] for r in passed)}, indent=1))
        print(f"baseline updated ({len(passed)} passing cases)")
        return 0
    if BASELINE.exists() and not args.filter:
        base = set(json.loads(BASELINE.read_text())["passed"])
        regressions = sorted(base - {r["description"] for r in passed})
        fixed = sorted({r["description"] for r in passed} - base)
        print(f"\nvs baseline: {len(regressions)} regression(s), {len(fixed)} newly passing")
        for d in regressions:
            print("  REGRESSION", d)
        return 1 if regressions else 0
    print("\n(no baseline yet: run with --update-baseline to save this run as the baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
