import re
from app.schemas import ChunkingStrategy


class TextChunker:
    """Implements multiple selectable text chunking strategies."""

    @staticmethod
    def chunk_text(
        text: str,
        strategy: ChunkingStrategy,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> list[str]:
        """Split raw text into chunks using the selected strategy."""
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return []

        if strategy == ChunkingStrategy.FIXED_WINDOW:
            return TextChunker._fixed_window_chunking(
                cleaned_text, chunk_size, chunk_overlap
            )
        return TextChunker._semantic_paragraph_chunking(text, max_chunk_size=chunk_size)

    @staticmethod
    def _fixed_window_chunking(
        text: str, chunk_size: int, chunk_overlap: int
    ) -> list[str]:
        """Strategy 1: Sliding character window with overlap preserving word boundaries."""
        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length:
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start = max(end - chunk_overlap, start + 1)

        return chunks

    @staticmethod
    def _semantic_paragraph_chunking(
        raw_text: str, max_chunk_size: int = 800
    ) -> list[str]:
        """Strategy 2: Semantic paragraph and sentence boundary aware chunking."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        chunks: list[str] = []
        current_buffer = ""

        for para in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_buffer) + len(sentence) + 1 <= max_chunk_size:
                    current_buffer = f"{current_buffer} {sentence}".strip()
                else:
                    if current_buffer:
                        chunks.append(current_buffer)
                    current_buffer = sentence

        if current_buffer:
            chunks.append(current_buffer)

        return chunks
