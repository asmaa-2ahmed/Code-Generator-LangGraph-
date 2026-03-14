from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

from src.config import (CHROMA_COLLECTION, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_ID, 
                        HUMANEVAL_PARQUET_URL, MAX_DISTANCE, RETRIEVAL_K )

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

def learn_new_function(function_name: str, code: str, explanation: str, original_query: str = "" ) -> str:
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
