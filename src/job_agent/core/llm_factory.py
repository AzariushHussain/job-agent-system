from typing import Type, TypeVar, Any
from pydantic import BaseModel
from src.job_agent.core.config import settings

T = TypeVar("T", bound=BaseModel)


def get_structured_llm(
    schema: Type[T],
    role: str = "fast",
) -> Any:
    """Returns an LLM bound with `.with_structured_output(schema)`."""
    model_name = (
        settings.writer_model_name if role == "writer" else settings.fast_model_name
    )

    if settings.llm_provider.lower() == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model_name,
            temperature=settings.temperature,
            api_key=settings.openai_api_key or None,
        )
    else:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=model_name,
            temperature=settings.temperature,
            api_key=settings.groq_api_key or None,
        )

    return llm.with_structured_output(schema)
