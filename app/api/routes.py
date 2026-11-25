import io
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import DocumentRecord, get_db
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkingStrategy,
    IngestionResponse,
)
from app.services.chunker import TextChunker
from app.services.memory import RedisChatMemory
from app.services.rag_agent import CustomConversationalRAG
from app.services.vector_store import QdrantVectorService

router = APIRouter(prefix="/api/v1", tags=["RAG & Ingestion APIs"])
settings = get_settings()

vector_service = QdrantVectorService()
memory_service = RedisChatMemory()
rag_service = CustomConversationalRAG(
    vector_service=vector_service, memory_service=memory_service
)


def _extract_text_from_upload(file: UploadFile, raw_bytes: bytes) -> str:
    """Extract UTF-8 text from an uploaded .pdf or .txt file."""
    filename = (file.filename or "").lower()
    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    if filename.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="ignore").strip()
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Only .pdf and .txt files are supported.",
    )


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Document Ingestion API",
)
async def ingest_document(
    file: UploadFile = File(..., description="Upload a .pdf or .txt document"),
    chunking_strategy: ChunkingStrategy = Form(
        default=ChunkingStrategy.SEMANTIC_PARAGRAPH,
        description="Select chunking strategy: fixed_window or semantic_paragraph",
    ),
    chunk_size: int = Form(default=600, ge=100, le=4000),
    chunk_overlap: int = Form(default=100, ge=0, le=1000),
    db: Session = Depends(get_db),
) -> IngestionResponse:
    """Ingest a PDF or TXT document, chunk text, store embeddings in Qdrant, and save metadata in SQL."""
    raw_bytes = await file.read()
    extracted_text = _extract_text_from_upload(file=file, raw_bytes=raw_bytes)
    if not extracted_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded document contains no extractable text.",
        )

    chunks = TextChunker.chunk_text(
        text=extracted_text,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not generate text chunks from document.",
        )

    document_id = str(uuid.uuid4())
    safe_filename = file.filename or "uploaded_document"

    stored_count = vector_service.upsert_chunks(
        document_id=document_id,
        filename=safe_filename,
        strategy=chunking_strategy.value,
        chunks=chunks,
    )

    doc_record = DocumentRecord(
        document_id=document_id,
        filename=safe_filename,
        content_type=file.content_type or "text/plain",
        chunking_strategy=chunking_strategy.value,
        total_chunks=stored_count,
        character_count=len(extracted_text),
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    return IngestionResponse(
        document_id=doc_record.document_id,
        filename=doc_record.filename,
        chunking_strategy=chunking_strategy,
        total_chunks_stored=doc_record.total_chunks,
        character_count=doc_record.character_count,
        vector_collection=settings.qdrant_collection,
        created_at=doc_record.created_at,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Conversational RAG & Interview Booking API",
)
def conversational_rag(
    payload: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Handle multi-turn RAG queries using Redis memory and book interviews via LLM tool calling."""
    return rag_service.run(
        session_id=payload.session_id,
        query=payload.query,
        top_k=payload.top_k,
        db=db,
    )
