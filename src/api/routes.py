from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    HealthResponse,
    LearnRequest,
    LearnResponse,
    QueryRequest,
    QueryResponse,
)
from src.graph.builder import learn_new_function, run_with_meta

router = APIRouter()


# ============================================================
# POST /query
# ============================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a prompt to the RAG code assistant",
    description=(
        "Classifies the prompt as **generate**, **explain**, or **self_learning**, "
        "routes it through the LangGraph pipeline, and returns the selected mode "
        "together with the assistant's response."
    ),
)
def query(body: QueryRequest) -> QueryResponse:
    """
    Main chat endpoint.

    - **generate** → retrieves similar HumanEval examples, then generates code.
    - **explain**  → retrieves context, then explains the code or concept.
    - **self_learning** → the query is unknown; asks the user to teach the system
      via POST /learn.
    """
    try:
        result = run_with_meta(
            user_query=body.prompt,
            thread_id=body.thread_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph execution failed: {exc}",
        ) from exc

    return QueryResponse(
        mode=result["mode"],
        response=result["response"],
        thread_id=result["thread_id"],
    )


# ============================================================
# POST /learn
# ============================================================

@router.post(
    "/learn",
    response_model=LearnResponse,
    status_code=status.HTTP_200_OK,
    summary="Teach the system a new function",
    description=(
        "Call this after receiving a **self_learning** response from POST /query. "
        "Provide the function name, source code, and explanation. "
        "The system embeds and stores it in Chroma so future similar queries "
        "will be answered automatically."
    ),
)
def learn(body: LearnRequest) -> LearnResponse:
    """Store a user-taught function in the vector store."""
    try:
        message = learn_new_function(
            function_name=body.function_name,
            code=body.code,
            explanation=body.explanation,
            thread_id=body.thread_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store function: {exc}",
        ) from exc

    return LearnResponse(message=message)


# ============================================================
# GET /health
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns 200 OK when the service is ready to accept requests.",
)
def health() -> HealthResponse:
    return HealthResponse()