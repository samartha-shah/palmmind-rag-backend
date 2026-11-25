from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "Palm Mind Conversational RAG & Ingestion API"
    llm_api_key: str = "sk-placeholder"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "palmmind_documents"
    qdrant_local_path: str = "./qdrant_storage"
    embedding_model: str = "all-MiniLM-L6-v2"

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 86400

    database_url: str = "sqlite:///./palmmind_metadata.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
