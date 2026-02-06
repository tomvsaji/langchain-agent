# LangChain Agent with Vector Memory

A full-stack conversational AI prototype featuring retrieval-augmented generation (RAG), persistent chat history, and agent memory. Built with **FastAPI**, **LangChain**, **ChromaDB**, **SQLite**, and **Streamlit**.

## Architecture

```
backend/
  config.py            # Shared configuration constants
  main.py              # FastAPI app with REST endpoints
  langchain_utils.py   # LangChain chain with RAG prompt template
  vector_store.py      # ChromaDB vector database wrapper
  chat_history.py      # SQLite chat history persistence
  agent_memory.py      # Agent memory combining history + retrieval
frontend/
  app.py               # Streamlit chat UI with session management
tests/
  conftest.py          # Shared pytest fixtures
  test_vector_store.py # Vector store unit tests
  test_chat_history.py # Chat history unit tests
  test_agent_memory.py # Agent memory unit tests
  test_langchain_utils.py # Chain integration tests
  test_api.py          # FastAPI endpoint integration tests
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Database** | ChromaDB | Stores document embeddings for similarity search |
| **Chat History** | SQLite | Persists conversation sessions and messages |
| **Agent Memory** | Custom module | Combines recent conversation history with retrieved knowledge to build LLM context |
| **LLM Chain** | LangChain Core | RAG pipeline: prompt template with knowledge + history slots piped to LLM |
| **API** | FastAPI | REST endpoints for chat, sessions, and document management |
| **Frontend** | Streamlit | Chat interface with session sidebar and knowledge base management |

### How It Works

1. User sends a message via the Streamlit UI (or API directly)
2. The **Agent Memory** module retrieves:
   - Recent conversation history from SQLite
   - Relevant document chunks from ChromaDB via similarity search
3. Both are injected into a LangChain prompt template alongside the user's message
4. The chain invokes the LLM (demo mock by default) with full context
5. The response and user message are persisted to SQLite for future turns

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the backend and frontend in separate terminals:

```bash
# Terminal 1 — API server
uvicorn backend.main:app --reload

# Terminal 2 — Streamlit UI
streamlit run frontend/app.py
```

The API runs on `http://localhost:8000` and the UI on `http://localhost:8501`.

## Run Tests

```bash
python -m pytest tests/ -v
```

All tests use temporary directories and are fully isolated — no persistent state is required.

## API Endpoints

### Chat

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a message. Creates a session if `session_id` is omitted. |

**Request body:**
```json
{"message": "What is RAG?", "session_id": null}
```

**Response:**
```json
{"response": "...", "session_id": "abc123..."}
```

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/sessions` | List all sessions (most recent first) |
| `GET` | `/sessions/{id}` | Get a single session |
| `GET` | `/sessions/{id}/messages` | Get all messages in a session |
| `DELETE` | `/sessions/{id}` | Delete a session and its messages |

### Documents (Vector Store)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents` | Add documents (auto-chunked) |
| `POST` | `/documents/search` | Similarity search |
| `POST` | `/documents/seed` | Load sample documents for demo |
| `GET` | `/documents/count` | Document count in the store |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with store/session stats |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000` | Backend URL for the Streamlit frontend |
| `DATA_DIR` | `data` | Directory for SQLite and ChromaDB storage |
| `CHROMA_COLLECTION` | `knowledge_base` | ChromaDB collection name |
| `RETRIEVAL_TOP_K` | `3` | Number of documents to retrieve per query |
| `MAX_MEMORY_TURNS` | `10` | Max conversation turns included in context |

## Swapping the Demo LLM for a Real One

The prototype uses a deterministic mock LLM (`_demo_llm` in `backend/langchain_utils.py`). To use a real LLM:

1. Install the provider package (e.g., `pip install langchain-openai`)
2. Replace the `RunnableLambda(_demo_llm)` in `build_chain()` with a real model:

```python
from langchain_openai import ChatOpenAI

def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_TEMPLATE),
        ("human", _HUMAN_TEMPLATE),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini")
    return prompt | llm | StrOutputParser()
```

3. Set your API key: `export OPENAI_API_KEY=sk-...`
