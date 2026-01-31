from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.langchain_utils import build_chain

app = FastAPI(title="LangChain FastAPI Demo")
chain = build_chain()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    response = chain.invoke({"message": request.message})
    return ChatResponse(response=response)
