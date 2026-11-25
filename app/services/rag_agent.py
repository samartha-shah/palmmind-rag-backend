import json
from typing import Any
from openai import OpenAI
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
                    "name": {
                        "type": "string",
                        "description": "Full name of the candidate or user",
                    },
                    "email": {
                        "type": "string",
                        "description": "Email address of the candidate",
                    },
                    "date": {
                        "type": "string",
                        "description": "Preferred interview date (e.g., 2026-11-28)",
                    },
                    "time": {
                        "type": "string",
                        "description": "Preferred interview time (e.g., 14:00 or 2:00 PM)",
                    },
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

    def _contextualize_query(
        self, query: str, history: list[dict[str, str]]
    ) -> str:
        """Rewrite a follow-up query into a standalone search query using chat history."""
        if not history:
            return query

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

        system_prompt = (
            "You are a helpful AI assistant for Palm Mind Technology. "
            "Answer the user's question accurately using the retrieved document context below. "
            "If the user asks to schedule or book an interview and provides their name, email, "
            "date, and time, invoke the `book_interview` tool. If they want to book an interview "
            "but are missing any of those four fields (name, email, date, time), politely ask "
            "for the missing details.\n\n"
            f"Retrieved Document Context:\n{context_block or 'No documents uploaded yet.'}"
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        booking_confirmation: BookingDetails | None = None

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
                booking_record = InterviewBooking(
                    session_id=session_id,
                    candidate_name=args["name"],
                    candidate_email=args["email"],
                    interview_date=args["date"],
                    interview_time=args["time"],
                )
                db.add(booking_record)
                db.commit()
                db.refresh(booking_record)

                booking_confirmation = BookingDetails(
                    booking_id=booking_record.id,
                    name=booking_record.candidate_name,
                    email=booking_record.candidate_email,
                    date=booking_record.interview_date,
                    time=booking_record.interview_time,
                )
                answer = (
                    f"Your interview has been successfully booked for {booking_record.candidate_name} "
                    f"({booking_record.candidate_email}) on {booking_record.interview_date} "
                    f"at {booking_record.interview_time} (Booking ID #{booking_record.id})."
                )
            else:
                answer = response_message.content or "Processed request."
        else:
            answer = response_message.content or "I could not generate a response."

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
