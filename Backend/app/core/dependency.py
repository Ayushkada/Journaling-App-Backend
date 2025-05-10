# app/core/dependency.py
from fastapi import Request
from functools import lru_cache
from app.analysis.ai_service import AIService
from app.analysis.local_ai_service import LocalAIService
from app.analysis.open_ai_service import OpenAIAIService

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
    model = request.query_params.get("model")
    if model == "chatgpt":
        return _chatgpt()
    if model == "local-large":
        return _local_large()
    return _local_small()
