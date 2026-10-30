"""RAG answer chain (LCEL): hybrid retrieve -> context (graph facts + chunks) -> prompt -> Haiku -> string.

The chain returns {"question", "docs", "graph_facts", "linked_entities", "answer"} so callers get the
retrieved chunks (for citations) without retrieving a second time.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import load_settings
from src.retrieval.hybrid import HybridRetriever

SYSTEM_PROMPT = """You are an assistant that answers employee questions about ACME Corp.

Rules:
- Answer ONLY from the context below. Do not use outside knowledge.
- After each claim, cite the chunk_id(s) it came from in square brackets, e.g. [SOP-17#2].
- Graph facts may be cited as [graph:<source_doc_id>].
- Technical manuals are authoritative for system dependencies; mention conflicting incident reports as conflicts.
- If the context does not contain the answer, say so explicitly instead of guessing.

Context:
{context}"""

MAX_OUTPUT_TOKENS = 4096  # the model spends part of this on reasoning before the visible answer

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


def format_context(graph_facts: list[str], docs: list[Document]) -> str:
    facts = "\n".join(f"- {f}" for f in graph_facts) or "(none)"
    passages = "\n\n".join(f"[{d.metadata['chunk_id']}]\n{d.page_content}" for d in docs)
    return f"Graph facts (from the knowledge graph):\n{facts}\n\nPassages:\n{passages}"


def build_answer_chain():
    settings = load_settings()
    retriever = HybridRetriever()
    llm = ChatAnthropic(
        model=settings.cheap_model, api_key=settings.anthropic_api_key, max_tokens=MAX_OUTPUT_TOKENS
    )

    def retrieve(question: str) -> dict:
        r = retriever.retrieve(question)
        return {
            "question": question,
            "docs": r["chunks"],
            "graph_facts": r["graph_facts"],
            "linked_entities": r["linked_entities"],
        }

    answer_path = (
        {"context": lambda x: format_context(x["graph_facts"], x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return RunnableLambda(retrieve) | RunnablePassthrough.assign(answer=answer_path)
