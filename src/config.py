# src/config.py
"""
Central configuration for the Self-Learning RAG Code Assistant.

Contains:
  - HuggingFace / OpenAI-compatible API settings
  - LLM model registry and factory
  - All system prompt strings
  - Tunable hyper-parameters (thresholds, limits)
"""

import os
from typing import Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()  # load .env file if present
# ============================================================
# API Settings
# ============================================================
BASE_URL: str = "https://router.huggingface.co/v1"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN is not set. "
        "Add it to the .env file ."
    )

# ============================================================
# Paths Settings
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(BASE_DIR, 'assets')

# ============================================================
# LLM Model Registry
# ============================================================
LLM_MODELS: dict = {
    "code": {
        "model":       "Qwen/Qwen3-Coder-480B-A35B-Instruct:hyperbolic",
        "temperature": 0.7,
    },
    "explain": {
        "model":       "Qwen/Qwen3-4B-Instruct-2507:nscale",
        "temperature": 0.7,
    },
    "self_learning": {  # intent classification + conversation summarisation
        "model":       "openai/gpt-oss-20b:groq",
        "temperature": 0.7,
    },
}

LLMRole = Literal["code", "explain", "self_learning"]


def make_llm(role: LLMRole) -> ChatOpenAI:
    """
    Instantiate a ChatOpenAI client for the given role.

    The ``self_learning`` role receives an extra ``tiktoken_model_name``
    so token-counting works correctly when used for summarisation.
    """
    cfg = LLM_MODELS[role]
    extra = (
        {"tiktoken_model_name": "gpt-3.5-turbo"} if role == "self_learning" else {}
    )
    return ChatOpenAI(
        model=cfg["model"],
        base_url=BASE_URL,
        api_key=HF_TOKEN,
        temperature=cfg["temperature"],
        **extra,
    )

# ============================================================
# System Prompts
# ============================================================
SYSTEM_CODE: str = """
You are a senior Python software engineer.
For each problem:
- Briefly analyse the root cause.
- Provide the correct solution.
- Always return the code inside a ```python``` block.
Do not add extra examples or explanations unless explicitly requested.
Keep responses concise, technical, friendly, slightly funny, and use emojis.
"""

SYSTEM_EXPLAIN: str = """
You are a senior Python developer.
Explain the given Python code or concept clearly and concisely.
If the input is not valid Python code, politely ask the user to provide code.
Keep it friendly, slightly funny, and use emojis.
"""

SYSTEM_INTENT: str = """
You are an intent classifier.
Classify the user request into exactly one word:
- generate   (user wants code written)
- explain    (user wants code or a concept explained)
- self_learning  (query is highly unusual, niche, or unknown domain)
Return only one word, nothing else.
"""

SYSTEM_SUMMARISE: str = """
Distil the conversation below into a single concise paragraph that
preserves the most important context (names, decisions, code produced).
"""

# ============================================================
# Tunable Hyper-parameters
# ============================================================
MAX_DISTANCE: float = 1.2    # L2 distance threshold for confidence gate
RETRIEVAL_K:  int   = 3      # number of similar docs to retrieve
SUMMARY_THRESHOLD: int = 8   # compress messages above this count
KEEP_RECENT: int = 4         # how many recent messages to keep after compression

# ============================================================
# Vector Store Settings
# ============================================================
EMBEDDING_MODEL_ID:    str = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION:     str = "code_generator_collection"
CHROMA_PERSIST_DIR:    str = os.path.join(assets_dir, "humaneval_db")
HUMANEVAL_PARQUET_URL: str = "hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"

# ============================================================
# Smoke Test
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("🔧  config.py — Smoke Test")
    print("=" * 55)

    # 1. Token present
    assert HF_TOKEN, "❌ HF_TOKEN is empty"
    print(f"✅  HF_TOKEN found  (length={len(HF_TOKEN)})")

    # 2. LLM factory — only check object type, don't call API
    for role in ("code", "explain", "self_learning"):
        llm = make_llm(role)          # type: ignore[arg-type]
        assert isinstance(llm, ChatOpenAI), f"❌  make_llm('{role}') returned wrong type"
        print(f"✅  make_llm('{role}') → {llm.model_name}")

    # 3. Prompts are non-empty strings
    for name, prompt in [
        ("SYSTEM_CODE",      SYSTEM_CODE),
        ("SYSTEM_EXPLAIN",   SYSTEM_EXPLAIN),
        ("SYSTEM_INTENT",    SYSTEM_INTENT),
        ("SYSTEM_SUMMARISE", SYSTEM_SUMMARISE),
    ]:
        assert isinstance(prompt, str) and prompt.strip(), f"❌  {name} is empty"
        print(f"✅  {name}  ({len(prompt.split())} words)")

    # 4. Numeric constants
    assert 0 < MAX_DISTANCE < 10
    assert RETRIEVAL_K > 0
    assert SUMMARY_THRESHOLD > 0
    print(f"✅  Hyper-params OK  (MAX_DISTANCE={MAX_DISTANCE}, K={RETRIEVAL_K})")

    print("\n🎉  config.py is healthy!")