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


def learn_new_function(function_name: str, code: str, explanation: str, thread_id: str = "default",) -> str:
    """
    Teach the system a new function and clear the pending_query state.

    Workflow
    --------
    1. Read ``pending_query`` from the graph's checkpoint for this thread.
    2. Call ``_store_function`` to embed + persist the document in Chroma.
    3. Clear ``pending_query`` in the checkpoint via ``app.update_state``.

    Parameters
    ----------
    function_name : str
        Python identifier for the new function (e.g. ``"quantum_entangle"``).
    code          : str
        Full function source code.
    explanation   : str
        Plain-text description of the function.
    thread_id     : str
        Must match the thread that triggered the ``self_learning`` response.

    Returns
    -------
    str
        Confirmation message.
    """
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


# # ============================================================
# # Smoke Test
# # ============================================================
# if __name__ == "__main__":
#     print("=" * 55)
#     print("🔧  builder.py — Smoke Test")
#     print("=" * 55)

#     # 1. Graph compiles without errors
#     assert app is not None
#     print("✅  Graph compiled successfully")

#     # 2. Graph structure (grandalf optional for ASCII rendering)
#     try:
#         ascii_graph = app.get_graph().draw_ascii()
#         assert len(ascii_graph) > 0
#         print("✅  ASCII graph rendered:")
#         print(ascii_graph)
#     except ImportError:
#         node_names = list(app.get_graph().nodes.keys())
#         print(f"✅  Graph nodes (install grandalf for ASCII art): {node_names}")

#     # 3. All expected nodes present
#     expected_nodes = {
#         "classify_intent", "generate", "retrieve",
#         "generate_code", "explain", "summarise",
#     }
#     actual_nodes = set(app.get_graph().nodes.keys()) - {"__start__", "__end__"}
#     missing = expected_nodes - actual_nodes
#     assert not missing, f"❌  Missing nodes: {missing}"
#     print(f"✅  All {len(expected_nodes)} nodes present in graph")

#     # 4. run() returns a string (calls the real graph / API)
#     try:
#         response = run(
#             "Write a Python function that checks if a number is even",
#             thread_id="smoke-test-thread",
#         )
#         assert isinstance(response, str) and len(response) > 0
#         print(f"✅  run() returned a response ({len(response)} chars)")
#         print(f"   Preview: {response[:200].replace(chr(10),' ')} ...")
#     except Exception as exc:
#         print(f"⚠️   run() skipped — API unreachable ({exc})")

#     print("\n🎉  builder.py is healthy!")