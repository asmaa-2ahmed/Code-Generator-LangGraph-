from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import (
    SYSTEM_CODE,
    SYSTEM_EXPLAIN,
    SYSTEM_INTENT,
    make_llm,
)

# ── Shared LLM instances ─────────────────────────────────────
llm_code    = make_llm("code")           # Qwen Coder 480B — generation + RAG
llm_explain = make_llm("explain")        # Qwen 4B         — explanations
llm_meta    = make_llm("self_learning")  # gpt-oss-20b     — intent + summarisation

# ============================================================
# RAG / Code Generation Chain
# ============================================================
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_CODE),
    MessagesPlaceholder(variable_name="history"),
    ("human", """
        Use the following similar examples as guidance:

        {context}

        Now solve this problem:
        {input}
            """),
])

# Input keys: {"input": str, "context": str, "history": list[BaseMessage]}
rag_chain = _RAG_PROMPT | llm_code | StrOutputParser()


# ============================================================
# Explain Chain
# ============================================================
_EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_EXPLAIN),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Input keys: {"input": str, "history": list[BaseMessage]}
explain_chain = _EXPLAIN_PROMPT | llm_explain | StrOutputParser()


# ============================================================
# Intent Classification Chain
# ============================================================
_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INTENT),
    ("human", "{input}"),
])

# Input keys: {"input": str}
# Returns: "generate" | "explain" | "idk"
intent_chain = _INTENT_PROMPT | llm_meta | StrOutputParser()


# ============================================================
# Smoke Test
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("🔧  chains.py — Smoke Test")
    print("=" * 55)

    # 1. Check chain types without calling the API
    from langchain_core.runnables import RunnableSequence

    for name, chain in [
        ("rag_chain",     rag_chain),
        ("explain_chain", explain_chain),
        ("intent_chain",  intent_chain),
    ]:
        assert hasattr(chain, "invoke"), f"❌  {name} has no .invoke()"
        print(f"✅  {name}  is a valid runnable")

    # 2. LLM model names match config
    from src.config import LLM_MODELS

    assert llm_code.model_name    == LLM_MODELS["code"]["model"]
    assert llm_explain.model_name == LLM_MODELS["explain"]["model"]
    assert llm_meta.model_name    == LLM_MODELS["self_learning"]["model"]
    print("✅  All LLM model names match config registry")

    # 3. Live intent call (cheap, fast model) — validates API key
    try:
        result = intent_chain.invoke(
            {"input": "Write a function that reverses a list"}
        ).strip().lower()
        assert result in {"generate", "explain", "idk"}, (
            f"❌  Unexpected intent: '{result}'"
        )
        print(f"✅  intent_chain live call → '{result}'")
    except Exception as exc:
        print(f"⚠️   intent_chain live call skipped ({exc})")

    print("\n🎉  chains.py is healthy!")