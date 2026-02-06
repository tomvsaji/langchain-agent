"""LangChain chain construction with retrieval-augmented generation and memory."""

from __future__ import annotations

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from backend.agent_memory import AgentMemory

_SYSTEM_TEMPLATE = """\
You are a helpful assistant with access to a knowledge base and conversation memory.

## Relevant knowledge
{knowledge}

## Conversation history
{history}

Use the knowledge and conversation history above to inform your answer. \
If the knowledge base has relevant information, incorporate it. \
Always be concise and accurate."""

_HUMAN_TEMPLATE = "{message}"


def _demo_llm(prompt_value: PromptValue) -> str:
    """A deterministic mock LLM for demo/testing purposes.

    Extracts the user question from the prompt and returns a formatted
    response that proves the retrieval context and history were wired in.
    """
    messages = prompt_value.to_messages()
    if not messages:
        return "I didn't receive a message. Could you try again?"

    system_content = messages[0].content if messages else ""
    user_content = messages[-1].content if len(messages) > 1 else ""

    # Determine if knowledge was provided
    has_knowledge = (
        "no relevant documents found" not in system_content.lower()
    )
    has_history = "no prior conversation" not in system_content.lower()

    parts: list[str] = [f"[Demo Agent] You asked: {user_content}"]
    if has_knowledge:
        parts.append("I found relevant information in the knowledge base.")
    if has_history:
        parts.append("I can see our prior conversation for context.")
    parts.append(
        "This is a demo response — swap in a real LLM for production use."
    )
    return " ".join(parts)


def build_chain() -> ChatPromptTemplate:
    """Build a LangChain runnable that accepts ``{message}``, ``{knowledge}``,
    and ``{history}`` and returns a string response."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_TEMPLATE),
            ("human", _HUMAN_TEMPLATE),
        ]
    )
    return prompt | RunnableLambda(_demo_llm)


def run_chain_with_memory(
    chain,
    memory: AgentMemory,
    session_id: str,
    message: str,
) -> str:
    """Invoke the chain with memory context and persist the turn."""
    context = memory.get_context(session_id, message)
    response = chain.invoke(
        {
            "message": message,
            "knowledge": context["knowledge"],
            "history": context["history"],
        }
    )
    memory.save_turn(session_id, message, response)
    return response
