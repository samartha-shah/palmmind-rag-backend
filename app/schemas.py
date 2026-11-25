from datetime import datetime
from enum import Enum
from pydantic import BaseModel, EmailStr, Field


class ChunkingStrategy(str, Enum):
    """Selectable text chunking strategies for Document Ingestion."""

    FIXED_WINDOW = "fixed_window"
    SEMANTIC_PARAGRAPH = "semantic_paragraph"


class IngestionResponse(BaseModel):
    """Response schema returned after document ingestion."""

    document_id: str
    filename: str
    chunking_strategy: ChunkingStrategy
    total_chunks_stored: int
    character_count: int
    vector_collection: str
    created_at: datetime


class ChatRequest(BaseModel):
    """Request payload for the Conversational RAG endpoint."""

    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique conversation session identifier for Redis memory",
    )
    query: str = Field(
        ..., min_length=1, description="User message or interview booking request"
    )
    top_k: int = Field(
        default=4, ge=1, le=15, description="Number of vector chunks to retrieve"
    )


class BookingDetails(BaseModel):
    """Structured interview booking details extracted by the LLM."""

    booking_id: int
    name: str
    email: EmailStr
    date: str
    time: str


class RetrievedChunk(BaseModel):
    """Metadata of a retrieved vector chunk used to ground the RAG response."""

    document_id: str
    filename: str
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    """Response schema for the Conversational RAG endpoint."""

    session_id: str
    answer: str
    reformulated_query: str
    retrieved_sources: list[RetrievedChunk]
    booking_confirmation: BookingDetails | None = None
