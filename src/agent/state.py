"""Agent state: what the graph nodes read and write."""
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The conversation (human, AI and tool messages). add_messages APPENDS what a node returns
    # instead of overwriting, which is what lets the agent<->tools loop build up its history.
    messages: Annotated[list[AnyMessage], add_messages]
    # Retrieval results for the latest question. Kept out of `messages` so that saved history
    # (per-session memory in Task 7) never carries stale context from earlier turns.
    chunks: list[Document]
    graph_facts: list[str]
    linked_entities: list[dict]
    steps: int  # agent-node calls so far this turn (reset by the retrieve node)
