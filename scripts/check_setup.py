"""Verify config loads and every external service is reachable. Prints PASS/FAIL, never secrets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings


def check_anthropic(s):
    from langchain_anthropic import ChatAnthropic

    ChatAnthropic(model=s.cheap_model, api_key=s.anthropic_api_key, max_tokens=5).invoke("Say hi")


def check_neo4j(s):
    from neo4j import GraphDatabase

    with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password)) as driver:
        driver.verify_connectivity()


def check_tavily(s):
    from tavily import TavilyClient

    TavilyClient(api_key=s.tavily_api_key).search("Halcyon", max_results=1)


def check_langfuse(s):
    from langfuse import Langfuse

    client = Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_host,
    )
    if not client.auth_check():
        raise RuntimeError("auth_check returned False")


def main() -> int:
    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"FAIL  config: {e}")
        return 1
    print("PASS  config")

    failed = 0
    for name, check in [
        ("anthropic", check_anthropic),
        ("neo4j", check_neo4j),
        ("tavily", check_tavily),
        ("langfuse", check_langfuse),
    ]:
        try:
            check(settings)
            print(f"PASS  {name}")
        except Exception as e:
            # Print only the exception type: messages can echo URLs or credentials.
            print(f"FAIL  {name} ({type(e).__name__})")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
