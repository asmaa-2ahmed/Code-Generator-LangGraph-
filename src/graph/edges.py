# src/graph/edges.py
"""
Conditional edge functions for the LangGraph workflow.

Each function receives the current ``AgentState`` and returns the
**name of the next node** as a plain string.  LangGraph uses these
strings as keys in the ``add_conditional_edges`` mapping.

Edge catalogue
--------------
route_after_retrieve
    Decides whether to ask the user to teach the system (self_learning),
    skip intent classification for user-taught docs (generate), or
    let the intent classifier decide (classify_intent).

route_after_intent
    Maps the classified intent string to the correct worker node.
"""

from __future__ import annotations

from src.graph.state import AgentState


# ============================================================
# Edge 1 — After intent_node
# ============================================================

def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "").strip().lower()
    if intent == "explain":
        return "explain"
    if intent == "self_learning":
        return "self_learning"
    return "generate"   # covers "generate" and any idk fallback


# ✅ ADD — after retrieve, branch by intent to the right code node
def route_after_retrieve(state: AgentState) -> str:
    intent = state.get("intent", "generate")
    if intent == "explain":
        return "explain_code"
    return "generate_code"
# ============================================================
# Smoke Test
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("🔧  edges.py — Smoke Test")
    print("=" * 55)

    base: AgentState = {
        "user_input":    "test",
        "messages":      [],
        "intent":        "",
        "context":       "",
        "response":      "",
        "known":         False,
        "doc_type":      "",
        "summary":       "",
        "pending_query": "",
    }

    # ── route_after_retrieve ──────────────────────────────────
    cases_retrieve = [
        # (known, doc_type, expected_next_node)
        (False, "",             "self_learning"),
        (True,  "user_taught",  "generate"),
        (True,  "humaneval",    "classify_intent"),
        (True,  "",             "classify_intent"),
    ]
    for known, doc_type, expected in cases_retrieve:
        state = {**base, "known": known, "doc_type": doc_type}
        result = route_after_retrieve(state)            # type: ignore[arg-type]
        assert result == expected, (
            f"❌  route_after_retrieve(known={known}, doc_type='{doc_type}') "
            f"→ '{result}', expected '{expected}'"
        )
        print(
            f"✅  route_after_retrieve "
            f"(known={known}, doc_type='{doc_type}') → '{result}'"
        )

    # ── route_after_intent ────────────────────────────────────
    cases_intent = [
        ("generate",    "generate"),
        ("explain",     "explain"),
        ("idk",         "unknown_intent"),
        ("gibberish",   "unknown_intent"),
        ("",            "unknown_intent"),
    ]
    for intent, expected in cases_intent:
        state = {**base, "intent": intent}
        result = route_after_intent(state)              # type: ignore[arg-type]
        assert result == expected, (
            f"❌  route_after_intent(intent='{intent}') "
            f"→ '{result}', expected '{expected}'"
        )
        print(f"✅  route_after_intent (intent='{intent}') → '{result}'")

    print("\n🎉  edges.py is healthy!")