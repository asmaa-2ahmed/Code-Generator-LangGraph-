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
