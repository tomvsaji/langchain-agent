"""Tests for the agent memory module."""

from backend.agent_memory import AgentMemory
from backend.chat_history import ChatHistoryDB
from backend.vector_store import VectorStore


class TestGetContext:
    def test_empty_context(self, agent_memory: AgentMemory, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Empty")
        ctx = agent_memory.get_context(sid, "hello")
        assert "no prior conversation" in ctx["history"]
        assert "no relevant documents found" in ctx["knowledge"]

    def test_context_includes_history(
        self, agent_memory: AgentMemory, chat_db: ChatHistoryDB
    ):
        sid = chat_db.create_session("With History")
        chat_db.add_message(sid, "human", "What is Python?")
        chat_db.add_message(sid, "ai", "A programming language.")
        ctx = agent_memory.get_context(sid, "Tell me more")
        assert "What is Python?" in ctx["history"]
        assert "A programming language." in ctx["history"]

    def test_context_includes_knowledge(
        self,
        agent_memory: AgentMemory,
        chat_db: ChatHistoryDB,
        vector_store: VectorStore,
    ):
        vector_store.add_documents(["FastAPI is a web framework for Python."])
        sid = chat_db.create_session("With Knowledge")
        ctx = agent_memory.get_context(sid, "What is FastAPI?")
        assert "FastAPI" in ctx["knowledge"]

    def test_context_with_both(
        self,
        agent_memory: AgentMemory,
        chat_db: ChatHistoryDB,
        vector_store: VectorStore,
    ):
        vector_store.add_documents(["Streamlit builds data apps."])
        sid = chat_db.create_session("Both")
        chat_db.add_message(sid, "human", "I like Streamlit")
        chat_db.add_message(sid, "ai", "Great choice!")
        ctx = agent_memory.get_context(sid, "Tell me about Streamlit")
        assert "Streamlit" in ctx["knowledge"]
        assert "I like Streamlit" in ctx["history"]


class TestSaveTurn:
    def test_save_turn_persists(
        self, agent_memory: AgentMemory, chat_db: ChatHistoryDB
    ):
        sid = chat_db.create_session("Turn Test")
        agent_memory.save_turn(sid, "Hi there", "Hello!")
        messages = chat_db.get_messages(sid)
        assert len(messages) == 2
        assert messages[0]["role"] == "human"
        assert messages[0]["content"] == "Hi there"
        assert messages[1]["role"] == "ai"
        assert messages[1]["content"] == "Hello!"

    def test_save_multiple_turns(
        self, agent_memory: AgentMemory, chat_db: ChatHistoryDB
    ):
        sid = chat_db.create_session("Multi Turn")
        agent_memory.save_turn(sid, "Q1", "A1")
        agent_memory.save_turn(sid, "Q2", "A2")
        messages = chat_db.get_messages(sid)
        assert len(messages) == 4


class TestMemoryFormatting:
    def test_history_format(
        self, agent_memory: AgentMemory, chat_db: ChatHistoryDB
    ):
        sid = chat_db.create_session("Format")
        chat_db.add_message(sid, "human", "hello")
        chat_db.add_message(sid, "ai", "hi")
        history = agent_memory._format_history(sid)
        assert "User: hello" in history
        assert "Assistant: hi" in history

    def test_knowledge_format(
        self, agent_memory: AgentMemory, vector_store: VectorStore
    ):
        vector_store.add_documents(["Test document content."])
        knowledge = agent_memory._retrieve_knowledge("test")
        assert "[1]" in knowledge
        assert "Test document content." in knowledge
