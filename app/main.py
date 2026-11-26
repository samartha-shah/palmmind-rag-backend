from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import get_settings
from app.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize SQL tables on startup."""
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Modular FastAPI Backend featuring Document Ingestion (Qdrant + SQL Metadata) "
        "and Custom Multi-Turn Conversational RAG with Redis Chat Memory and LLM Interview Booking."
    ),
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

init_db()
