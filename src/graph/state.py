from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
# from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    # ── Per-turn working fields ───────────────────────────────
    user_input:    str   # raw query from the user
    intent:        str   # "generate" | "explain" | "idk"
    context:       str   # formatted RAG context block
    response:      str   # final answer text

    # ── Confidence gate ───────────────────────────────────────
    known:    bool  # True  = distance < MAX_DISTANCE
    doc_type: str   # "user_taught" | "humaneval" | ""

    # ── Long-term memory ──────────────────────────────────────
    summary:       str   # compressed summary of older messages
    pending_query: str   # stored when the system enters learning mode


# # ============================================================
# # Smoke Test
# # ============================================================
# if __name__ == "__main__":
#     print("=" * 55)
#     print("🔧  state.py — Smoke Test")
#     print("=" * 55)

#     from langchain_core.messages import AIMessage, HumanMessage

#     # 1. AgentState is a valid TypedDict — construct a minimal instance
#     state: AgentState = {
#         "user_input": "Write a palindrome checker",
#         "messages":   [HumanMessage(content="Hello")],
#         "intent":     "",
#         "context":    "",
#         "response":   "",
#         "known":      False,
#         "doc_type":   "",
#         "summary":    "",
#         "pending_query": "",
#     }

#     assert state["user_input"] == "Write a palindrome checker"
#     assert len(state["messages"]) == 1
#     print("✅  AgentState construction OK")

#     # 2. add_messages reducer merges lists correctly
#     from langgraph.graph.message import add_messages

#     existing = [HumanMessage(content="Hi")]
#     incoming = [AIMessage(content="Hello!")]
#     merged = add_messages(existing, incoming)
#     assert len(merged) == 2
#     print(f"✅  add_messages reducer → merged {len(merged)} messages")

#     # 3. All expected keys are present in the annotation
#     expected_keys = {
#         "messages", "user_input", "intent", "context", "response",
#         "known", "doc_type", "summary", "pending_query",
#     }
#     actual_keys = set(AgentState.__annotations__.keys())
#     missing = expected_keys - actual_keys
#     assert not missing, f"❌  Missing keys: {missing}"
#     print(f"✅  All {len(expected_keys)} state keys present")

#     print("\n🎉  state.py is healthy!")