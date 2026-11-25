# Conversational RAG & Document Ingestion Backend (FastAPI)

A modular, fully-typed **FastAPI** backend built for the **Palm Mind Technology AI/ML Intern Assessment**.

## Architecture & Features
1. **Document Ingestion API (`POST /api/v1/ingest`)**
   - Supports `.pdf` and `.txt` file uploads.
   - **Two Selectable Chunking Strategies**:
     - `fixed_window`: Sliding character window with overlap preserving word boundaries.
     - `semantic_paragraph`: Paragraph and sentence boundary-aware semantic chunking.
   - Generates dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and stores vectors + payload in **Qdrant**.
   - Saves document metadata in a **SQL Database** (`document_metadata` table) via SQLAlchemy 2.0.

2. **Conversational RAG & Interview Booking API (`POST /api/v1/chat`)**
   - **Custom RAG Pipeline** (built from scratch without `RetrievalQAChain`, FAISS, or Chroma).
   - **Redis Chat Memory**: Stores and retrieves multi-turn session history keyed by `session_id`.
   - **Multi-Turn Query Contextualization**: Rewrites follow-up queries using recent conversation history before querying Qdrant.
   - **LLM Interview Booking**: Uses structured LLM Function/Tool Calling (`book_interview`) to extract `name`, `email`, `date`, and `time` from conversation and stores bookings in the SQL database (`interview_bookings` table).

## Quickstart
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your LLM_API_KEY in .env
uvicorn app.main:app --reload
```
Open **`http://127.0.0.1:8000/docs`** to test both APIs via Swagger UI.
