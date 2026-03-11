# src/memory/vectorstore.py
"""
Vector store + retrieval layer.

Responsibilities
----------------
- Lazy singletons for the embedding model and Chroma vector store.
- Load the HumanEval benchmark into Chroma (one-shot setup).
- Confidence-gated similarity search for the retrieve node.
- Store user-taught functions for the self-learning flow.
- Build RAG context strings consumed by the generate node.
"""

from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_ID,
    HUMANEVAL_PARQUET_URL,
    MAX_DISTANCE,
    RETRIEVAL_K,
)

# ============================================================
# Lazy Singletons
# ============================================================
_embedding_model: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return (and cache) the shared HuggingFace embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={"device": "cpu"},
        )
    return _embedding_model


def get_vectorstore() -> Chroma:
    """Return (and cache) the Chroma vector store instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embedding_model(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


# ============================================================
# Data Ingestion
# ============================================================

def load_humaneval() -> None:
    """Load the HumanEval benchmark from HuggingFace and push it into Chroma."""
    global _vectorstore

    df = pd.read_parquet(HUMANEVAL_PARQUET_URL)
    print(f"📦  HumanEval — shape={df.shape}, duplicates={df.duplicated().sum()}")

    documents = [
        Document(
            page_content=row["prompt"],
            metadata={
                "task_id":     row["task_id"],
                "solution":    row["canonical_solution"],
                "entry_point": row["entry_point"],
            },
        )
        for _, row in df.iterrows()
    ]

    _vectorstore = Chroma.from_documents(
        collection_name=CHROMA_COLLECTION,
        documents=documents,
        embedding=get_embedding_model(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"✅  Loaded {len(documents)} HumanEval documents into Chroma.")


# ============================================================
# Retrieval
# ============================================================

def get_retriever():
    """Return a configured LangChain retriever over the vector store."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )


def retrieve_with_confidence(query: str) -> tuple[Document | None, bool, str]:
    """
    Single-shot similarity search with a confidence gate.

    Returns
    -------
    doc      : The closest Document (or None if empty collection).
    known    : True when L2 distance < MAX_DISTANCE.
    doc_type : "user_taught" | "humaneval" | ""
    """
    results = get_vectorstore().similarity_search_with_score(query, k=1)

    if not results:
        print("   🔬 [DEBUG] Vectorstore returned no results")
        return None, False, ""

    doc, score = results[0]
    print(f"   🔬 [DEBUG] L2 distance: {score:.4f} | threshold: {MAX_DISTANCE}")

    if score >= MAX_DISTANCE:
        return None, False, ""

    doc_type = doc.metadata.get("type", "humaneval")
    return doc, True, doc_type


# ============================================================
# Self-Learning
# ============================================================

def learn_new_function(
    function_name: str,
    code: str,
    explanation: str,
    original_query: str = "",
) -> str:
    """Embed and store a user-taught function in Chroma."""
    label = original_query or function_name
    document_text = (
        f"Query: {label}\n\n"
        f"Function: {function_name}\n\n"
        f"Code:\n{code}\n\n"
        f"Explanation:\n{explanation}"
    )
    new_doc = Document(
        page_content=document_text,
        metadata={
            "type":           "user_taught",
            "function_name":  function_name,
            "original_query": label,
        },
    )
    get_vectorstore().add_documents([new_doc])
    return f"✅ Learned and stored '{function_name}' in the vector store 🚀"


# ============================================================
# Context Builders
# ============================================================

def build_rag_context(query: str) -> str:
    """Retrieve top-k similar documents and format them as a RAG context block."""
    docs = get_retriever().invoke(query)
    return "\n\n".join(
        f"Prompt:\n{doc.page_content}\n"
        f"Solution:\n{doc.metadata.get('solution', doc.page_content)}"
        for doc in docs
    )


def build_taught_context(doc: Document) -> str:
    """Format a single user-taught document as a context block."""
    return f"Learned function:\n{doc.page_content}"


# # ============================================================
# # Smoke Test
# # ============================================================
# if __name__ == "__main__":
#     print("=" * 55)
#     print("🔧  vectorstore.py — Smoke Test")
#     print("=" * 55)

#     # 1. Embedding model
#     em = get_embedding_model()
#     test_embed = em.embed_query("hello world")
#     assert len(test_embed) > 0, "❌  Embedding returned empty vector"
#     print(f"✅  Embedding model OK  (dim={len(test_embed)})")

#     # 2. Vectorstore instantiated
#     vs = get_vectorstore()
#     assert vs is not None
#     print("✅  Chroma vectorstore instantiated")

#     # 3. Add a dummy doc + search
#     dummy = Document(
#         page_content="def palindrome(s): return s == s[::-1]",
#         metadata={"solution": "return s == s[::-1]", "type": "smoke_test"},
#     )
#     vs.add_documents([dummy])
#     hits = vs.similarity_search("check palindrome", k=1)
#     assert hits, "❌  similarity_search returned nothing"
#     print(f"✅  Similarity search OK  → '{hits[0].page_content[:50]}...'")

#     # 4. Confidence gate
#     doc, known, doc_type = retrieve_with_confidence("check palindrome")
#     print(f"✅  retrieve_with_confidence → known={known}, doc_type='{doc_type}'")

#     # 5. learn_new_function
#     msg = learn_new_function(
#         function_name="smoke_fn",
#         code="def smoke_fn(): pass",
#         explanation="Smoke test placeholder.",
#         original_query="write a smoke test function",
#     )
#     assert "smoke_fn" in msg
#     print(f"✅  learn_new_function → {msg}")

#     # 6. Retriever helper
#     ret = get_retriever()
#     assert hasattr(ret, "invoke")
#     docs = ret.invoke("add two numbers")
#     assert isinstance(docs, list)
#     print(f"✅  get_retriever().invoke() → {len(docs)} docs")

#     # 7. Context builders
#     ctx = build_rag_context("check palindrome")
#     assert isinstance(ctx, str) and len(ctx) > 0
#     print(f"✅  build_rag_context → {len(ctx)} chars")

#     taught_doc = Document(
#         page_content="Query: ...\nFunction: foo\nCode:\ndef foo(): pass\nExplanation: nothing",
#         metadata={"type": "user_taught", "function_name": "foo"},
#     )
#     taught_ctx = build_taught_context(taught_doc)
#     assert "Learned function:" in taught_ctx and "foo" in taught_ctx
#     print(f"✅  build_taught_context → {len(taught_ctx)} chars")

#     print("\n🎉  vectorstore.py is healthy!")