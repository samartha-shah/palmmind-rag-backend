import json
import redis
from redis.exceptions import RedisError

from app.config import get_settings

settings = get_settings()


class RedisChatMemory:
    """Manages multi-turn conversational history in Redis with safe fallback."""

    def __init__(self) -> None:
        self._fallback_store: dict[str, list[dict[str, str]]] = {}
        try:
            self.client: redis.Redis | None = redis.Redis.from_url(
                settings.redis_url, decode_responses=True, socket_connect_timeout=2
            )
            self.client.ping()
        except RedisError:
            self.client = None

    def _key(self, session_id: str) -> str:
        return f"chat_memory:{session_id}"

    def get_history(self, session_id: str, max_turns: int = 8) -> list[dict[str, str]]:
        """Retrieve recent conversation turns for the given session_id."""
        key = self._key(session_id)
        if self.client is not None:
            try:
                raw_messages = self.client.lrange(key, -max_turns * 2, -1)
                return [json.loads(item) for item in raw_messages]
            except RedisError:
                pass
        return self._fallback_store.get(key, [])[-max_turns * 2 :]

    def append_turn(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Append user and assistant messages to Redis session history."""
        key = self._key(session_id)
        entries = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        if self.client is not None:
            try:
                for entry in entries:
                    self.client.rpush(key, json.dumps(entry))
                self.client.expire(key, settings.redis_ttl_seconds)
                return
            except RedisError:
                pass

        self._fallback_store.setdefault(key, []).extend(entries)
