from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.edges import route_after_intent,route_after_retrieve
from src.graph.nodes import (
    explain_code_node,
    generate_code_node,
    intent_node,
    retrieve_node,
    self_learning_node,
    summarise_node,
)
from src.graph.state import AgentState
from src.memory.vectorstore import learn_new_function as _store_function

# ============================================================
# Build the Workflow
# ============================================================

def _build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────
    workflow.add_node("classify_intent", intent_node)
    workflow.add_node("retrieve",        retrieve_node)
    workflow.add_node("generate_code",   generate_code_node)
    workflow.add_node("explain_code",    explain_code_node)
    workflow.add_node("self_learning",   self_learning_node)
    workflow.add_node("summarise",       summarise_node)

    workflow.set_entry_point("classify_intent")

    # classify_intent → one of three branches
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "generate":      "retrieve",
            "explain":       "retrieve",
            "self_learning": "self_learning",
        },
    )

    # after retrieve → generate_code or explain_code based on intent
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "generate_code": "generate_code",
            "explain_code":  "explain_code",
        },
    )

    # all paths converge at summarise → END
    workflow.add_edge("generate_code",  "summarise")
    workflow.add_edge("explain_code",   "summarise")
    workflow.add_edge("self_learning",  "summarise")
    workflow.add_edge("summarise",      END)
    return workflow


# Compile once at import time with an in-process MemorySaver
_workflow   = _build_graph()
_checkpointer = MemorySaver()
app = _workflow.compile(checkpointer=_checkpointer)

# ============================================================
# Public API
# ============================================================

def run(user_query: str, thread_id: str = "default") -> str:

    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_input": user_query}, config=config)
    return result["response"]

def run_with_meta(user_query: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_input": user_query}, config=config)
    return {
        "mode":      result.get("intent", "generate"),
        "response":  result.get("response", ""),
        "thread_id": thread_id,
    }

def learn_new_function(function_name: str, code: str, explanation: str, thread_id: str = "default",) -> str:
    
    config = {"configurable": {"thread_id": thread_id}}

    # Pull the pending query from checkpoint state (may be empty string)
    snapshot = app.get_state(config)
    pending  = snapshot.values.get("pending_query", "") or function_name

    # Embed and store the new document
    msg = _store_function(
        function_name=function_name,
        code=code,
        explanation=explanation,
        original_query=pending,
    )

    # Clear pending_query in the graph's checkpoint
    app.update_state(config, {"pending_query": ""})

    return msg

