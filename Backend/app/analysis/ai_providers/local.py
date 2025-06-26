from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from app.analysis.ai_providers.base import AIService
from sentence_transformers import SentenceTransformer
import torch


class LocalAIService(AIService):
    def __init__(self):
        super().__init__()
        self.model_tag = "local-small"

        device = -1  # CPU only

        # Sentiment Analysis
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=device
        )

        # Emotion Classification
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device
        )

        # T5-based Generation
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base", torch_dtype=torch.float32
        )

        self.t5_model = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        # Embedding Model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
