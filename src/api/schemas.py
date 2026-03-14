from __future__ import annotations
from pydantic import BaseModel, Field


# ============================================================
# /query
# ============================================================

class QueryRequest(BaseModel):
    """Body accepted by POST /query."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The user's question or code task.",
        examples=["Write a function that reverses a string"],
    )
    thread_id: str = Field(
        default="default",
        description=(
            "Conversation identifier. Use the same value across turns to "
            "keep memory continuity. Use a unique value to start a fresh session."
        ),
        examples=["user-42-session"],
    )


class QueryResponse(BaseModel):
    """Body returned by POST /query."""

    mode: str = Field(
        description="Intent selected by the classifier: 'generate', 'explain', or 'self_learning'.",
        examples=["generate"],
    )
    response: str = Field(
        description="The assistant's final answer for this turn.",
    )
    thread_id: str = Field(
        description="Echo of the thread_id used, so clients can track it.",
    )


# ============================================================
# /learn
# ============================================================

class LearnRequest(BaseModel):
    """Body accepted by POST /learn."""

    function_name: str = Field(
        ...,
        description="Python identifier for the function being taught.",
        examples=["quantum_entangle"],
    )
    code: str = Field(
        ...,
        description="Full function source code.",
        examples=["def quantum_entangle(a, b):\n    return (a + b) / 2"],
    )
    explanation: str = Field(
        ...,
        description="Plain-text description of what the function does.",
        examples=["Simulates quantum entanglement by averaging two qubit states."],
    )
    thread_id: str = Field(
        default="default",
        description="Must match the thread that triggered the self-learning response.",
    )


class LearnResponse(BaseModel):
    """Body returned by POST /learn."""

    message: str = Field(
        description="Confirmation that the function was stored.",
    )


# ============================================================
# /health
# ============================================================

class HealthResponse(BaseModel):
    """Body returned by GET /health."""

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")