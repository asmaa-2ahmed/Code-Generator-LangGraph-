from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import KEEP_RECENT, SUMMARY_THRESHOLD, SYSTEM_SUMMARISE

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

    from src.graph.state import AgentState


# ============================================================
# History Builder
# ============================================================

def build_history(state: "AgentState") -> List[BaseMessage]:

    msgs: List[BaseMessage] = state.get("messages", [])
    summary: str = state.get("summary", "")

    if len(msgs) > SUMMARY_THRESHOLD and summary:
        return (
            [SystemMessage(content=f"Conversation summary so far:\n{summary}")]
            + msgs[-KEEP_RECENT:]
        )
    return msgs


# ============================================================
# Summarisation
# ============================================================

def summarise_if_needed(state: "AgentState", llm_meta: "ChatOpenAI") -> dict:
    
    msgs: List[BaseMessage] = state.get("messages", [])

    if len(msgs) <= SUMMARY_THRESHOLD:
        return {}  
    
    to_compress = msgs[:-KEEP_RECENT]
    conversation_text = "\n".join(
        f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in to_compress
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_SUMMARISE),
        ("human", "{conversation}"),
    ])
    chain = prompt | llm_meta | StrOutputParser()
    new_summary = chain.invoke({"conversation": conversation_text})

    return {
        "summary":  new_summary,
        "messages": msgs[-KEEP_RECENT:],
    }
