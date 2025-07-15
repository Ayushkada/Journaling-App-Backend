from fastapi import Request
from functools import lru_cache
from Backend.app.analysis.ai_providers.base import AIService
from Backend.app.analysis.ai_providers.local import LocalAIService
from Backend.app.analysis.ai_providers.openai import OpenAIAIService
import logging

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "local-small"

@lru_cache(maxsize=None)
def _local_small() -> AIService:
    return LocalAIService()

@lru_cache(maxsize=None)
def _local_large() -> AIService:
    return LocalAIService(large=True)

@lru_cache(maxsize=None)
def _chatgpt() -> AIService:
    return OpenAIAIService()

def get_ai_service(request: Request) -> AIService:
    """
    FastAPI dependency that returns the appropriate AI service implementation
    based on the `model` query parameter.
    """
    model = request.query_params.get("model", DEFAULT_MODEL).strip().lower()

    if model == "chatgpt":
        return _chatgpt()
    if model == "local-large":
        return _local_large()

    if model != DEFAULT_MODEL:
        logger.warning(f"Unknown model '{model}' requested. Falling back to default '{DEFAULT_MODEL}'.")

    return _local_small()
