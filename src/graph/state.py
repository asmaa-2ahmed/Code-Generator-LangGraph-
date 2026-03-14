from __future__ import annotations
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    user_input:    str   # raw query from the user
    intent:        str   # "generate" | "explain" | "idk"
    context:       str   # formatted RAG context block
    response:      str   # final answer text

    known:    bool  # True  = distance < MAX_DISTANCE
    doc_type: str   # "user_taught" | "humaneval" | ""

    summary:       str   # compressed summary of older messages
    pending_query: str   # stored when the system enters learning mode
