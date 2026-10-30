"""Ask the RAG chain a question: python scripts/ask.py "your question"."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.answer_chain import build_answer_chain


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python scripts/ask.py "your question"')
    result = build_answer_chain().invoke(sys.argv[1])
    print(f"Q: {result['question']}\n")
    print(result["answer"])
    print("\nLinked entities:", ", ".join(e["name"] for e in result["linked_entities"]) or "(none)")
    print("Retrieved chunk_ids:", ", ".join(d.metadata["chunk_id"] for d in result["docs"]))
    print("Graph facts in context:", len(result["graph_facts"]))


if __name__ == "__main__":
    main()
