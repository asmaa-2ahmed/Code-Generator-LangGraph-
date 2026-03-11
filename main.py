# main.py
"""
Self-Learning RAG Code Assistant — main entry point.

Run from the project root:
    python main.py

What happens
------------
1. Ingest the HumanEval dataset into Chroma (first-run only).
2. Print the graph topology.
3. Run the full end-to-end integration test suite.
"""

import sys

# ── Setup ─────────────────────────────────────────────────────
# Ensure the project root is on sys.path when running with
# `python main.py` instead of `python -m main`
import os
# sys.path.insert(0, os.path.dirname(__file__))

from src.memory.vectorstore import load_humaneval
from src.graph.builder import app, run, learn_new_function

# ============================================================
# Step 1 — Ingest HumanEval (skip if already populated)
# ============================================================
def _maybe_ingest() -> None:
    """Ingest HumanEval only when the collection is empty."""
    from src.memory.vectorstore import get_vectorstore

    count = get_vectorstore()._collection.count()   # Chroma internal count
    if count == 0:
        print("📦  Vector store is empty — ingesting HumanEval dataset …")
        load_humaneval()
    else:
        print(f"📦  Vector store already has {count} documents — skipping ingest.")


# ============================================================
# Step 2 — Print graph topology
# ============================================================
def _print_graph() -> None:
    print("\n" + "=" * 60)
    print("🗺️   LangGraph Topology")
    print("=" * 60)
    try:
        print(app.get_graph().draw_ascii())
    except ImportError:
        nodes = list(app.get_graph().nodes.keys())
        print(f"Nodes: {nodes}")
        print("(Install grandalf for ASCII art: pip install grandalf)")


# ============================================================
# Step 3 — Integration Tests
# ============================================================
THREAD = "main-integration-test"

_SEPARATOR = "─" * 60


def _section(title: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {title}")
    print(_SEPARATOR)


def run_integration_tests() -> None:
    _section("Test 1 — Memory continuity (introduce user)")
    resp = run("My name is Asmaa and I am a backend developer.", THREAD)
    print(resp)

    _section("Test 2 — Known query from HumanEval (palindrome check)")
    resp = run("Write a function to check if a string is a palindrome", THREAD)
    print(resp)

    _section("Test 3 — Unknown query → triggers self-learning mode")
    resp = run("Write a quantum entanglement simulation function", THREAD)
    print(resp)
    assert "🤔" in resp, "Expected self-learning trigger response"

    _section("Teaching the system a new function")
    msg = learn_new_function(
        function_name="quantum_entangle",
        code="""
def quantum_entangle(qubit_a, qubit_b):
    # Simplified Bell state simulation
    return (qubit_a + qubit_b) / 2
""",
        explanation=(
            "Simulates basic quantum entanglement by averaging two qubit states."
        ),
        thread_id=THREAD,
    )
    print(msg)
    assert "✅" in msg

    _section("Test 4 — Same query AFTER learning (should use stored knowledge)")
    resp = run("Write a quantum entanglement simulation function", THREAD)
    print(resp)

    _section("Test 5 — Memory: recall user name and profession")
    resp = run("What is my name and what do I do?", THREAD)
    print(resp)

    _section("Test 6 — Explain intent")
    resp = run("Explain how list comprehensions work in Python", THREAD)
    print(resp)

    _section("Test 7 — Memory-aware follow-up (improve last function)")
    resp = run("Can you improve the last function you generated?", THREAD)
    print(resp)

    _section("Test 8 — Unknown intent graceful fallback")
    # Force idk by sending an ambiguous non-query
    resp = run("asdfghjkl qwerty 12345", THREAD)
    print(resp)

    print(f"\n{'=' * 60}")
    print("🎉  All integration tests completed!")
    print(f"{'=' * 60}\n")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀  Self-Learning RAG Code Assistant")
    print("=" * 60)

    _maybe_ingest()
    _print_graph()
    run_integration_tests()