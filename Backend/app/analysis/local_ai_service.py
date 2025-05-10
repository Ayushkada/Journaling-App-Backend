from transformers import pipeline
from app.analysis.ai_service import AIService
from sentence_transformers import SentenceTransformer

class LocalAIService(AIService):
    def __init__(self, large: bool = False):
        super().__init__()                 
        self.model_tag = "local-large" if large else "local-small"

        if large:
            self.sentiment_model = pipeline("text-classification", model="roberta-large-go-emotions")
            self.emotion_model = pipeline("text-classification",   model="microsoft/deberta-v3-large-emotion")
            self.t5_model = pipeline("text2text-generation",  model="google/flan-t5-large")
            self.embedding_model = SentenceTransformer("all-mpnet-base-v2")