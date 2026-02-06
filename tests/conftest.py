"""Shared pytest fixtures for all test modules."""

import tempfile
from pathlib import Path

import pytest

from backend.chat_history import ChatHistoryDB
from backend.vector_store import VectorStore
from backend.agent_memory import AgentMemory


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory (provided by pytest)."""
    return tmp_path


@pytest.fixture()
def chat_db(tmp_dir: Path) -> ChatHistoryDB:
    """Return a ChatHistoryDB backed by a temp SQLite file."""
    return ChatHistoryDB(db_path=str(tmp_dir / "test_chat.db"))


@pytest.fixture()
def vector_store(tmp_dir: Path) -> VectorStore:
    """Return a VectorStore backed by a temp ChromaDB directory."""
    return VectorStore(
        persist_directory=str(tmp_dir / "test_chroma"),
        collection_name="test_collection",
    )


@pytest.fixture()
def agent_memory(chat_db: ChatHistoryDB, vector_store: VectorStore) -> AgentMemory:
    """Return an AgentMemory wired to the temp chat DB and vector store."""
    return AgentMemory(chat_db=chat_db, vector_store=vector_store)
