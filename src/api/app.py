from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.memory.vectorstore import get_vectorstore, load_humaneval


# ============================================================
# Lifespan — runs once on startup and once on shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀  Starting up RAG Code Assistant API …")

    vs = get_vectorstore()
    count = vs._collection.count()
    if count == 0:
        print("📦  Vector store is empty — loading HumanEval dataset …")
        load_humaneval()
        print("✅  HumanEval loaded.")
    else:
        print(f"📦  Vector store already has {count} documents — skipping ingest.")

    yield  # API is live and serving requests here

    print("👋  Shutting down RAG Code Assistant API.")


# ============================================================
# Application Factory
# ============================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Code Assistant",
        description=(
            "A self-learning RAG pipeline powered by LangGraph. "
            "Send a prompt, get generated code or an explanation back. "
            "Teach the system new functions it doesn't know yet."
        ),
        version="1.0.0",
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc UI
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1", tags=["assistant"])

    return app


app = create_app()
