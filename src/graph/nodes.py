from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import AgentState
from src.memory.conversation import build_history, summarise_if_needed
from src.memory.vectorstore import retrieve_with_confidence
from src.rag.chains import explain_chain, intent_chain, llm_meta, rag_chain
from src.memory.vectorstore import build_rag_context, build_taught_context

# ============================================================
# Node 1 — Retrieve with Confidence
# ============================================================

def retrieve_node(state: AgentState) -> dict:
    query = state["user_input"]
    doc, known, doc_type = retrieve_with_confidence(query)

    if not known:
        return {"known": False, "doc_type": "", "context": ""}

    # Build context differently depending on document source
    context = (
        build_taught_context(doc)
        if doc_type == "user_taught"
        else build_rag_context(query)
    )

    return {"known": True, "doc_type": doc_type, "context": context}


# ============================================================
# Node 2 — Classify Intent
# ============================================================

def intent_node(state: AgentState) -> dict:
    intent = intent_chain.invoke({"input": state["user_input"]}).strip().lower()
    print(f"   🎯 [Intent] → '{intent}'")
    return {"intent": intent}


# ============================================================
# Node 3 — Generate Code (RAG-aware)
# ============================================================

# generate_code: call the RAG chain with retrieved context
def generate_code_node(state: AgentState) -> dict:
    """Generate code using rag_chain + the context fetched by retrieve_node."""
    response = rag_chain.invoke({
        "input":   state["user_input"],
        "context": state.get("context", ""),
        "history": build_history(state),
    })
    return {
        "response": response,
        "messages": [
            HumanMessage(content=state["user_input"]),
            AIMessage(content=response),
        ],
    }

# ============================================================
# Node 4 — Explain Code
# ============================================================

def explain_code_node(state: AgentState) -> dict:
    """Explain code or a concept using explain_chain + RAG context."""
    response = explain_chain.invoke({
        "input":   state["user_input"],
        "history": build_history(state),
    })
    return {
        "response": response,
        "messages": [
            HumanMessage(content=state["user_input"]),
            AIMessage(content=response),
        ],
    }


# ============================================================
# Node 5 — Self-Learning (unknown query)
# ============================================================

def self_learning_node(state: AgentState) -> dict:
    response = """
        🤔 I don't know this yet!
        Can you teach me the correct solution? Please provide:
          1. Code
          2. Explanation
          3. Example usage
        """
    return {
        "response":      response,
        "pending_query": state["user_input"],
        "messages": [
            HumanMessage(content=state["user_input"]),
            AIMessage(content=response),
        ],
    }


# ============================================================
# Node 6 — Unknown Intent Fallback
# ============================================================

def unknown_intent_node(state: AgentState) -> dict:
    intent = state.get("intent", "?")
    response = f"❓ Couldn't classify intent (got: '{intent}'). Try rephrasing!"
    return {
        "response": response,
        "messages": [
            HumanMessage(content=state["user_input"]),
            AIMessage(content=response),
        ],
    }


# ============================================================
# Node 7 — Summarise Conversation
# ============================================================

def summarise_node(state: AgentState) -> dict:
    return summarise_if_needed(state, llm_meta)


# # ============================================================
# # Smoke Test
# # ============================================================
# if __name__ == "__main__":
#     print("=" * 55)
#     print("🔧  nodes.py — Smoke Test")
#     print("=" * 55)

#     # All node functions must be callable and accept a dict-like state
#     nodes = [
#         retrieve_node,
#         intent_node,
#         generate_node,
#         explain_node,
#         self_learning_node,
#         unknown_intent_node,
#         summarise_node,
#     ]
#     for fn in nodes:
#         assert callable(fn), f"❌  {fn.__name__} is not callable"
#         print(f"✅  {fn.__name__}  is callable")

#     # self_learning_node — pure logic, no API call needed
#     test_state: AgentState = {
#         "user_input": "Write a quantum teleportation function",
#         "messages":   [],
#         "intent":     "",
#         "context":    "",
#         "response":   "",
#         "known":      False,
#         "doc_type":   "",
#         "summary":    "",
#         "pending_query": "",
#     }
#     result = self_learning_node(test_state)
#     assert result["pending_query"] == test_state["user_input"]
#     assert "🤔" in result["response"]
#     assert len(result["messages"]) == 2
#     print("✅  self_learning_node → correct state updates (no API call)")

#     # unknown_intent_node — pure logic
#     test_state["intent"] = "idk"
#     result = unknown_intent_node(test_state)
#     assert "idk" in result["response"]
#     assert len(result["messages"]) == 2
#     print("✅  unknown_intent_node → correct state updates (no API call)")

#     # summarise_node — short buffer → no-op
#     result = summarise_node(test_state)
#     assert result == {}, "❌  Expected no-op for short buffer"
#     print("✅  summarise_node (short buffer) → no-op ✓")

#     print("\n🎉  nodes.py is healthy!")