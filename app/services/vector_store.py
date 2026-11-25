import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from app.config import get_settings

settings = get_settings()


class QdrantVectorService:
    """Manages embedding generation and vector search in Qdrant."""

    def __init__(self) -> None:
        self.encoder = SentenceTransformer(settings.embedding_model)
        self.vector_size: int = (
            self.encoder.get_sentence_embedding_dimension() or 384
        )
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key
            )
        else:
            self.client = QdrantClient(path=settings.qdrant_local_path)

        self.collection_name = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = [
            col.name for col in self.client.get_collections().collections
        ]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        strategy: str,
        chunks: list[str],
    ) -> int:
        """Embed document chunks and store them in Qdrant with metadata payload."""
        embeddings = self.encoder.encode(chunks, normalize_embeddings=True).tolist()
        points: list[qmodels.PointStruct] = []

        for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{idx}"))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": idx,
                        "chunking_strategy": strategy,
                        "text": chunk_text,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Perform semantic similarity search against Qdrant."""
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        retrieved: list[dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            retrieved.append(
                {
                    "document_id": str(payload.get("document_id", "")),
                    "filename": str(payload.get("filename", "")),
                    "text": str(payload.get("text", "")),
                    "score": round(float(hit.score), 4),
                }
            )
        return retrieved
