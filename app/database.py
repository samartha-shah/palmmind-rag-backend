from datetime import datetime, timezone
from typing import Generator
from sqlalchemy import DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings

settings = get_settings()

connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


class DocumentRecord(Base):
    """SQL model storing ingested document metadata."""

    __tablename__ = "document_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class InterviewBooking(Base):
    """SQL model storing LLM-extracted interview bookings."""

    __tablename__ = "interview_bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    candidate_name: Mapped[str] = mapped_column(String(255), nullable=False)
    candidate_email: Mapped[str] = mapped_column(String(255), nullable=False)
    interview_date: Mapped[str] = mapped_column(String(64), nullable=False)
    interview_time: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def init_db() -> None:
    """Create database tables on application startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a SQL database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
