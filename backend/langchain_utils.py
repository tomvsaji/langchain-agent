from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


def _demo_llm(prompt_value: PromptValue) -> str:
    messages = prompt_value.to_messages()
    if not messages:
        return "Demo response: (no message provided)"
    return f"Demo response: {messages[-1].content}"


def build_chain() -> RunnableLambda:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that answers succinctly."),
            ("human", "{message}"),
        ]
    )
    return prompt | RunnableLambda(_demo_llm)
