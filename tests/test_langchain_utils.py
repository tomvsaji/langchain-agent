"""Tests for the LangChain chain utilities."""

from pathlib import Path

import pytest

from backend.agent_memory import AgentMemory
from backend.chat_history import ChatHistoryDB
from backend.langchain_utils import build_chain, run_chain_with_memory
from backend.vector_store import VectorStore


class TestBuildChain:
    def test_chain_returns_string(self):
        chain = build_chain()
        result = chain.invoke(
            {
                "message": "hello",
                "knowledge": "(no relevant documents found)",
                "history": "(no prior conversation)",
            }
        )
        assert isinstance(result, str)
        assert "hello" in result

    def test_chain_mentions_knowledge_when_present(self):
        chain = build_chain()
        result = chain.invoke(
            {
                "message": "What is RAG?",
                "knowledge": "[1] RAG combines retrieval with generation.",
                "history": "(no prior conversation)",
            }
        )
        assert "knowledge base" in result.lower()

    def test_chain_mentions_history_when_present(self):
        chain = build_chain()
        result = chain.invoke(
            {
                "message": "Continue",
                "knowledge": "(no relevant documents found)",
                "history": "User: hello\nAssistant: hi there",
            }
        )
        assert "conversation" in result.lower()

    def test_chain_demo_label(self):
        chain = build_chain()
        result = chain.invoke(
            {
                "message": "test",
                "knowledge": "(no relevant documents found)",
                "history": "(no prior conversation)",
            }
        )
        assert "[Demo Agent]" in result


class TestRunChainWithMemory:
    def test_run_persists_turn(self, tmp_path: Path):
        db = ChatHistoryDB(db_path=str(tmp_path / "chain_chat.db"))
        vs = VectorStore(
            persist_directory=str(tmp_path / "chain_chroma"),
            collection_name="chain_test",
        )
        mem = AgentMemory(chat_db=db, vector_store=vs)
        chain = build_chain()
        sid = db.create_session("Chain Test")

        result = run_chain_with_memory(chain, mem, sid, "What is AI?")
        assert isinstance(result, str)

        messages = db.get_messages(sid)
        assert len(messages) == 2
        assert messages[0]["content"] == "What is AI?"
        assert messages[1]["content"] == result

    def test_run_uses_prior_history(self, tmp_path: Path):
        db = ChatHistoryDB(db_path=str(tmp_path / "hist_chat.db"))
        vs = VectorStore(
            persist_directory=str(tmp_path / "hist_chroma"),
            collection_name="hist_test",
        )
        mem = AgentMemory(chat_db=db, vector_store=vs)
        chain = build_chain()
        sid = db.create_session("History Test")

        run_chain_with_memory(chain, mem, sid, "First message")
        result = run_chain_with_memory(chain, mem, sid, "Second message")
        # Second call should see history
        assert "conversation" in result.lower()
