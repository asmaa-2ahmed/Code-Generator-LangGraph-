from __future__ import annotations

from src.graph.state import AgentState


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "").strip().lower()
    if intent == "explain":
        return "explain"
    if intent == "self_learning":
        return "self_learning"
    return "generate"   # covers "generate" and any idk fallback


def route_after_retrieve(state: AgentState) -> str:
    intent = state.get("intent", "generate")
    if intent == "explain":
        return "explain_code"
    return "generate_code"
