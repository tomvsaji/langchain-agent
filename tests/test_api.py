"""Integration tests for the FastAPI endpoints."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.agent_memory import AgentMemory
from backend.chat_history import ChatHistoryDB
from backend.vector_store import VectorStore


@pytest.fixture()
def client(tmp_path: Path):
    """Create a test client with isolated storage."""
    # Build isolated dependencies
    db = ChatHistoryDB(db_path=str(tmp_path / "api_chat.db"))
    vs = VectorStore(
        persist_directory=str(tmp_path / "api_chroma"),
        collection_name="api_test",
    )
    mem = AgentMemory(chat_db=db, vector_store=vs)

    # Import and patch the app's singletons
    from backend import main as main_module
    from backend.langchain_utils import build_chain

    main_module.chat_db = db
    main_module.vector_store = vs
    main_module.memory = mem
    main_module.chain = build_chain()

    return TestClient(main_module.app)


class TestHealth:
    def test_health_check(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "vector_store_documents" in data
        assert "sessions" in data


class TestChat:
    def test_chat_creates_session(self, client: TestClient):
        resp = client.post("/chat", json={"message": "Hello!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "session_id" in data
        assert len(data["session_id"]) == 32

    def test_chat_continues_session(self, client: TestClient):
        r1 = client.post("/chat", json={"message": "First"})
        sid = r1.json()["session_id"]
        r2 = client.post("/chat", json={"message": "Second", "session_id": sid})
        assert r2.status_code == 200
        assert r2.json()["session_id"] == sid

    def test_chat_invalid_session(self, client: TestClient):
        resp = client.post(
            "/chat", json={"message": "Hi", "session_id": "nonexistent"}
        )
        assert resp.status_code == 404

    def test_chat_empty_message_rejected(self, client: TestClient):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_chat_response_reflects_knowledge(self, client: TestClient):
        # Seed knowledge, then ask about it
        client.post("/documents/seed")
        resp = client.post("/chat", json={"message": "What is LangChain?"})
        data = resp.json()
        assert "knowledge base" in data["response"].lower()


class TestSessions:
    def test_list_sessions_empty(self, client: TestClient):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_sessions_after_chat(self, client: TestClient):
        client.post("/chat", json={"message": "Hello"})
        resp = client.get("/sessions")
        assert len(resp.json()) == 1

    def test_get_session(self, client: TestClient):
        r = client.post("/chat", json={"message": "Hello"})
        sid = r.json()["session_id"]
        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sid

    def test_get_session_not_found(self, client: TestClient):
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 404

    def test_get_session_messages(self, client: TestClient):
        r = client.post("/chat", json={"message": "Tell me a story"})
        sid = r.json()["session_id"]
        resp = client.get(f"/sessions/{sid}/messages")
        assert resp.status_code == 200
        msgs = resp.json()
        assert len(msgs) == 2  # human + ai
        assert msgs[0]["role"] == "human"
        assert msgs[1]["role"] == "ai"

    def test_delete_session(self, client: TestClient):
        r = client.post("/chat", json={"message": "Bye"})
        sid = r.json()["session_id"]
        resp = client.delete(f"/sessions/{sid}")
        assert resp.status_code == 200
        assert client.get(f"/sessions/{sid}").status_code == 404

    def test_delete_session_not_found(self, client: TestClient):
        resp = client.delete("/sessions/nonexistent")
        assert resp.status_code == 404


class TestDocuments:
    def test_add_documents(self, client: TestClient):
        resp = client.post(
            "/documents", json={"texts": ["Some knowledge to store."]}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert len(data["ids"]) >= 1

    def test_document_count(self, client: TestClient):
        resp = client.get("/documents/count")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

        client.post("/documents", json={"texts": ["data"]})
        resp = client.get("/documents/count")
        assert resp.json()["count"] >= 1

    def test_search_documents(self, client: TestClient):
        client.post(
            "/documents",
            json={"texts": ["Python is a great programming language"]},
        )
        resp = client.post(
            "/documents/search", json={"query": "Python", "top_k": 1}
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert "Python" in results[0]["document"]

    def test_search_empty_store(self, client: TestClient):
        resp = client.post(
            "/documents/search", json={"query": "anything", "top_k": 3}
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_seed_documents(self, client: TestClient):
        resp = client.post("/documents/seed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["seeded"] >= 5

    def test_add_with_metadata(self, client: TestClient):
        resp = client.post(
            "/documents",
            json={
                "texts": ["Metadata test"],
                "metadatas": [{"source": "unit_test"}],
            },
        )
        assert resp.status_code == 200
