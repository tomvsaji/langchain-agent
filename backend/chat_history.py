"""Chat history persistence using SQLite."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.config import CHAT_HISTORY_DB

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('human', 'ai', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class ChatHistoryDB:
    """Thin wrapper around SQLite for storing chat sessions and messages."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = str(db_path or CHAT_HISTORY_DB)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_SESSIONS)
            conn.execute(_CREATE_MESSAGES)

    # -- sessions ----------------------------------------------------------

    def create_session(self, title: str = "New Chat") -> str:
        """Create a new chat session and return its ID."""
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        return session_id

    def list_sessions(self) -> list[dict]:
        """Return all sessions ordered by most recently updated."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    # -- messages ----------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """Append a message to a session and return the message ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> list[dict]:
        """Return messages for a session, oldest first."""
        query = "SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC"
        params: tuple = (session_id,)
        if limit:
            query += " LIMIT ?"
            params = (session_id, limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_recent_messages(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """Return the most recent *limit* messages for a session, oldest first."""
        query = """
            SELECT * FROM (
                SELECT id, session_id, role, content, created_at
                FROM messages WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
            ) sub ORDER BY id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(query, (session_id, limit)).fetchall()
        return [dict(r) for r in rows]
