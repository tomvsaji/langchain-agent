"""FastAPI application with chat, sessions, and document management."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agent_memory import AgentMemory
from backend.chat_history import ChatHistoryDB
from backend.langchain_utils import build_chain, run_chain_with_memory
from backend.vector_store import VectorStore

app = FastAPI(title="LangChain Agent API")

# --- singletons initialised at startup -----------------------------------

chat_db = ChatHistoryDB()
vector_store = VectorStore()
memory = AgentMemory(chat_db=chat_db, vector_store=vector_store)
chain = build_chain()


# --- request / response models -------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(
        default=None,
        description="Optional session ID. A new session is created when omitted.",
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class DocumentAddRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    metadatas: list[dict] | None = None


class DocumentAddResponse(BaseModel):
    ids: list[str]
    count: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class SearchResult(BaseModel):
    document: str
    metadata: dict
    distance: float


# --- health ---------------------------------------------------------------


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "vector_store_documents": vector_store.count(),
        "sessions": len(chat_db.list_sessions()),
    }


# --- chat -----------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id
    if not session_id:
        session_id = chat_db.create_session(title=request.message[:60])
    elif not chat_db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    response = run_chain_with_memory(chain, memory, session_id, request.message)
    return ChatResponse(response=response, session_id=session_id)


# --- sessions -------------------------------------------------------------


@app.get("/sessions", response_model=list[SessionOut])
def list_sessions() -> list[dict]:
    return chat_db.list_sessions()


@app.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str) -> dict:
    session = chat_db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def get_session_messages(session_id: str) -> list[dict]:
    if not chat_db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_db.get_messages(session_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if not chat_db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


# --- documents (vector store) --------------------------------------------


@app.post("/documents", response_model=DocumentAddResponse)
def add_documents(request: DocumentAddRequest) -> DocumentAddResponse:
    ids = vector_store.add_documents(
        texts=request.texts, metadatas=request.metadatas
    )
    return DocumentAddResponse(ids=ids, count=len(ids))


@app.post("/documents/search", response_model=list[SearchResult])
def search_documents(request: SearchRequest) -> list[SearchResult]:
    hits = vector_store.similarity_search(request.query, top_k=request.top_k)
    return [SearchResult(**h) for h in hits]


@app.post("/documents/seed")
def seed_documents() -> dict:
    ids = vector_store.seed_sample_documents()
    return {"seeded": len(ids), "ids": ids}


@app.get("/documents/count")
def document_count() -> dict:
    return {"count": vector_store.count()}
