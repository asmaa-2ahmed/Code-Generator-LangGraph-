import os
from typing import Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv() 

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


# ====  System Prompts  =============================================
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

# ====  Parameters  ========================================
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
