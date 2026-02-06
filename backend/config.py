"""Application configuration constants."""

import os
from pathlib import Path

# Base data directory for persistent storage
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database for chat history
CHAT_HISTORY_DB = DATA_DIR / "chat_history.db"

# ChromaDB persistent storage
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")

# ChromaDB collection name
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")

# Number of documents to retrieve for context
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

# Maximum conversation turns to include in agent memory context
MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "10"))
