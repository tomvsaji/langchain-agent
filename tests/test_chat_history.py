"""Tests for the chat history SQLite module."""

from backend.chat_history import ChatHistoryDB


class TestSessionLifecycle:
    def test_create_session_returns_id(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Test Session")
        assert isinstance(sid, str)
        assert len(sid) == 32  # uuid hex

    def test_list_sessions_empty(self, chat_db: ChatHistoryDB):
        assert chat_db.list_sessions() == []

    def test_list_sessions_returns_created(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Session A")
        sessions = chat_db.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid
        assert sessions[0]["title"] == "Session A"

    def test_get_session(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Lookup")
        session = chat_db.get_session(sid)
        assert session is not None
        assert session["title"] == "Lookup"

    def test_get_session_not_found(self, chat_db: ChatHistoryDB):
        assert chat_db.get_session("nonexistent") is None

    def test_delete_session(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("To Delete")
        assert chat_db.delete_session(sid) is True
        assert chat_db.get_session(sid) is None

    def test_delete_session_not_found(self, chat_db: ChatHistoryDB):
        assert chat_db.delete_session("nonexistent") is False

    def test_sessions_ordered_by_updated(self, chat_db: ChatHistoryDB):
        s1 = chat_db.create_session("First")
        s2 = chat_db.create_session("Second")
        # Touch s1 by adding a message
        chat_db.add_message(s1, "human", "hello")
        sessions = chat_db.list_sessions()
        # s1 should now be first (most recently updated)
        assert sessions[0]["id"] == s1


class TestMessages:
    def test_add_and_get_messages(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Chat")
        mid1 = chat_db.add_message(sid, "human", "Hi")
        mid2 = chat_db.add_message(sid, "ai", "Hello!")
        assert isinstance(mid1, int)
        assert isinstance(mid2, int)

        messages = chat_db.get_messages(sid)
        assert len(messages) == 2
        assert messages[0]["role"] == "human"
        assert messages[0]["content"] == "Hi"
        assert messages[1]["role"] == "ai"
        assert messages[1]["content"] == "Hello!"

    def test_get_messages_empty_session(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Empty")
        assert chat_db.get_messages(sid) == []

    def test_get_messages_with_limit(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Limited")
        for i in range(5):
            chat_db.add_message(sid, "human", f"msg {i}")
        messages = chat_db.get_messages(sid, limit=3)
        assert len(messages) == 3

    def test_get_recent_messages(self, chat_db: ChatHistoryDB):
        sid = chat_db.create_session("Recent")
        for i in range(10):
            chat_db.add_message(sid, "human", f"msg {i}")
        recent = chat_db.get_recent_messages(sid, limit=3)
        assert len(recent) == 3
        # Should be the last 3 messages, in order
        assert recent[0]["content"] == "msg 7"
        assert recent[1]["content"] == "msg 8"
        assert recent[2]["content"] == "msg 9"

    def test_messages_isolated_between_sessions(self, chat_db: ChatHistoryDB):
        s1 = chat_db.create_session("A")
        s2 = chat_db.create_session("B")
        chat_db.add_message(s1, "human", "for A")
        chat_db.add_message(s2, "human", "for B")
        assert len(chat_db.get_messages(s1)) == 1
        assert len(chat_db.get_messages(s2)) == 1
        assert chat_db.get_messages(s1)[0]["content"] == "for A"
