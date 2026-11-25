import json
import re
from typing import Any
from openai import OpenAI, OpenAIError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import InterviewBooking
from app.schemas import BookingDetails, ChatResponse, RetrievedChunk
from app.services.memory import RedisChatMemory
from app.services.vector_store import QdrantVectorService

settings = get_settings()

BOOKING_TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "book_interview",
            "description": (
                "Book an interview when the user provides their name, email, "
                "preferred date, and preferred time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the candidate"},
                    "email": {"type": "string", "description": "Email address of the candidate"},
                    "date": {"type": "string", "description": "Preferred interview date (e.g., 2026-11-28)"},
                    "time": {"type": "string", "description": "Preferred interview time (e.g., 14:00 or 2:00 PM)"},
                },
                "required": ["name", "email", "date", "time"],
            },
        },
    }
]


class CustomConversationalRAG:
    """Custom RAG engine with multi-turn query contextualization and LLM interview booking."""

    def __init__(
        self, vector_service: QdrantVectorService, memory_service: RedisChatMemory
    ) -> None:
        self.vector_service = vector_service
        self.memory = memory_service
        self.llm = OpenAI(
            api_key=settings.llm_api_key, base_url=settings.llm_base_url
        )

    def _has_valid_key(self) -> bool:
        return bool(
            settings.llm_api_key
            and settings.llm_api_key not in {"sk-placeholder", "your_api_key_here"}
        )

    def _contextualize_query(
        self, query: str, history: list[dict[str, str]]
    ) -> str:
        """Rewrite a follow-up query into a standalone search query using chat history."""
        if not history:
            return query

        if not self._has_valid_key():
            last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
            return f"{last_user_msgs[-1]} {query}".strip() if last_user_msgs else query

        recent_context = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history[-4:]
        )
        prompt = (
            "Given the recent conversation history and the latest user question, "
            "rewrite the user question into a concise standalone search query for "
            "vector retrieval. Return ONLY the rewritten query text.\n\n"
            f"Conversation History:\n{recent_context}\n\n"
            f"Latest User Question: {query}"
        )
        try:
            completion = self.llm.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            rewritten = completion.choices[0].message.content
            return rewritten.strip() if rewritten else query
        except Exception:
            return query

    def _save_booking(
        self, db: Session, session_id: str, name: str, email: str, date: str, time: str
    ) -> tuple[BookingDetails, str]:
        booking_record = InterviewBooking(
            session_id=session_id,
            candidate_name=name,
            candidate_email=email,
            interview_date=date,
            interview_time=time,
        )
        db.add(booking_record)
        db.commit()
        db.refresh(booking_record)

        details = BookingDetails(
            booking_id=booking_record.id,
            name=booking_record.candidate_name,
            email=booking_record.candidate_email,
            date=booking_record.interview_date,
            time=booking_record.interview_time,
        )
        msg = (
            f"Your interview has been booked for {booking_record.candidate_name} "
            f"({booking_record.candidate_email}) on {booking_record.interview_date} "
            f"at {booking_record.interview_time} (Booking ID #{booking_record.id})."
        )
        return details, msg

    def _fallback_parse_and_answer(
        self, session_id: str, query: str, raw_hits: list[dict[str, Any]], db: Session
    ) -> tuple[str, BookingDetails | None]:
        """Fallback handler when running locally without an external LLM API key."""
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", query)
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", query)
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|\d{1,2}\s*(?:AM|PM|am|pm))\b", query)
        name_match = re.search(r"(?:name is|for|I am|I'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", query)

        if email_match and (date_match or time_match):
            name = name_match.group(1) if name_match else "Candidate"
            email = email_match.group(0)
            date = date_match.group(1) if date_match else "2026-11-28"
            time = time_match.group(1) if time_match else "10:00 AM"
            details, msg = self._save_booking(db, session_id, name, email, date, time)
            return msg, details

        if raw_hits:
            top_context = " ".join(hit["text"] for hit in raw_hits[:2])
            return f"Based on the ingested documents: {top_context}", None
        return "No relevant documents found yet. Please upload a .pdf or .txt file via /api/v1/ingest.", None

    def run(
        self, session_id: str, query: str, top_k: int, db: Session
    ) -> ChatResponse:
        """Execute multi-turn retrieval, LLM tool evaluation, and response generation."""
        history = self.memory.get_history(session_id=session_id)
        standalone_query = self._contextualize_query(query=query, history=history)

        raw_hits = self.vector_service.search(query=standalone_query, top_k=top_k)
        context_block = "\n\n".join(
            f"[Source {idx + 1} - {hit['filename']}]: {hit['text']}"
            for idx, hit in enumerate(raw_hits)
        )

        booking_confirmation: BookingDetails | None = None

        if not self._has_valid_key():
            answer, booking_confirmation = self._fallback_parse_and_answer(
                session_id, query, raw_hits, db
            )
        else:
            system_prompt = (
                "You are a helpful AI assistant for Palm Mind Technology. "
                "Answer the user's question accurately using the retrieved document context below. "
                "If the user asks to schedule or book an interview and provides their name, email, "
                "date, and time, invoke the `book_interview` tool.\n\n"
                f"Retrieved Document Context:\n{context_block or 'No documents uploaded yet.'}"
            )
            messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": query})

            try:
                response = self.llm.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    tools=BOOKING_TOOL_SCHEMA,
                    tool_choice="auto",
                    temperature=0.2,
                )
                response_message = response.choices[0].message
                if response_message.tool_calls:
                    tool_call = response_message.tool_calls[0]
                    if tool_call.function.name == "book_interview":
                        args = json.loads(tool_call.function.arguments)
                        booking_confirmation, answer = self._save_booking(
                            db,
                            session_id,
                            args["name"],
                            args["email"],
                            args["date"],
                            args["time"],
                        )
                    else:
                        answer = response_message.content or "Processed request."
                else:
                    answer = response_message.content or "I could not generate a response."
            except OpenAIError:
                answer, booking_confirmation = self._fallback_parse_and_answer(
                    session_id, query, raw_hits, db
                )

        self.memory.append_turn(
            session_id=session_id, user_message=query, assistant_message=answer
        )

        sources = [
            RetrievedChunk(
                document_id=hit["document_id"],
                filename=hit["filename"],
                score=hit["score"],
                text_preview=hit["text"][:200],
            )
            for hit in raw_hits
        ]

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            reformulated_query=standalone_query,
            retrieved_sources=sources,
            booking_confirmation=booking_confirmation,
        )
