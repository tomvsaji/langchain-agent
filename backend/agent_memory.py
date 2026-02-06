"""Agent memory module that combines conversation history with vector retrieval."""

from __future__ import annotations

from typing import Optional

from backend.chat_history import ChatHistoryDB
from backend.config import MAX_MEMORY_TURNS, RETRIEVAL_TOP_K
from backend.vector_store import VectorStore


class AgentMemory:
    """Builds a context window for the LLM by combining:

    1. **Conversation history** – the most recent turns from the chat session.
    2. **Retrieved knowledge** – relevant chunks from the vector store.
    """

    def __init__(
        self,
        chat_db: ChatHistoryDB,
        vector_store: VectorStore,
        max_turns: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.chat_db = chat_db
        self.vector_store = vector_store
        self.max_turns = max_turns or MAX_MEMORY_TURNS
        self.top_k = top_k or RETRIEVAL_TOP_K

    def get_context(self, session_id: str, current_query: str) -> dict:
        """Return a dict with ``history`` and ``knowledge`` strings.

        ``history``
            Formatted recent conversation turns.
        ``knowledge``
            Relevant document chunks retrieved from the vector store.
        """
        history = self._format_history(session_id)
        knowledge = self._retrieve_knowledge(current_query)
        return {"history": history, "knowledge": knowledge}

    def _format_history(self, session_id: str) -> str:
        messages = self.chat_db.get_recent_messages(
            session_id, limit=self.max_turns * 2  # each turn = human + ai
        )
        if not messages:
            return "(no prior conversation)"
        lines: list[str] = []
        for msg in messages:
            prefix = "User" if msg["role"] == "human" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    def _retrieve_knowledge(self, query: str) -> str:
        hits = self.vector_store.similarity_search(query, top_k=self.top_k)
        if not hits:
            return "(no relevant documents found)"
        parts: list[str] = []
        for i, hit in enumerate(hits, 1):
            parts.append(f"[{i}] {hit['document']}")
        return "\n".join(parts)

    def save_turn(
        self, session_id: str, user_message: str, ai_response: str
    ) -> None:
        """Persist a human/AI turn to the chat history database."""
        self.chat_db.add_message(session_id, "human", user_message)
        self.chat_db.add_message(session_id, "ai", ai_response)
