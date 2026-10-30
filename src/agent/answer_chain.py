"""Basic RAG answer chain (LCEL): retrieve -> format with chunk_ids -> prompt -> Haiku -> string.

The outer chain returns {"question", "docs", "answer"} so callers get the retrieved chunks
(for citations) without retrieving a second time.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.config import TOP_K, load_settings
from src.retrieval.vector import get_vector_store

SYSTEM_PROMPT = """You are an assistant that answers employee questions about ACME Corp.

Rules:
- Answer ONLY from the context below. Do not use outside knowledge.
- After each claim, cite the chunk_id(s) it came from in square brackets, e.g. [SOP-17#2].
- If the context does not contain the answer, say so explicitly instead of guessing.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


def format_chunks(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['chunk_id']}]\n{d.page_content}" for d in docs)


def build_answer_chain():
    settings = load_settings()
    retriever = get_vector_store().as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatAnthropic(
        model=settings.cheap_model, api_key=settings.anthropic_api_key, max_tokens=1024
    )

    answer_path = (
        {"context": lambda x: format_chunks(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return (
        {"question": RunnablePassthrough(), "docs": retriever}
        | RunnablePassthrough.assign(answer=answer_path)
    )
