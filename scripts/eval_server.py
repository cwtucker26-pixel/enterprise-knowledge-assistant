"""Start the real API (uvicorn) for an evaluation run and stop it afterwards.

    from scripts.eval_server import serve
    with serve() as base_url:        # e.g. http://127.0.0.1:8765
        ...call base_url + "/chat"...

Evals go through HTTP so they test what a user would get: the real lifespan, guards, sessions and tracing.
Server output goes to a log file (never the terminal), so secrets cannot end up in eval transcripts.
"""
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "eval" / "results" / "logs"
DEFAULT_PORT = 8765


@contextmanager
def serve(port: int = DEFAULT_PORT, startup_timeout: int = 240):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"server_{port}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"the API exited during startup (see {LOG_DIR / f'server_{port}.log'})")
            try:
                if httpx.get(base + "/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("the API did not become healthy in time")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":  # python scripts/eval_server.py  -> run a server in the foreground until Ctrl-C
    with serve() as url:
        print("API up at", url, "(Ctrl-C to stop)")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
