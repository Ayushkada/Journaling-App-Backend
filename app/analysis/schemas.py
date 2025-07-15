# schemas.py
from typing import List, Dict, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        orm_mode = True


class AnalysisRequest(BaseSchema):
    journals: List["JournalEntryCreate"]  # Use forward references if needed
    goals: Optional[List["GoalCreate"]] = []


class JournalAnalysisBase(BaseSchema):
    id: UUID
    journal_id: UUID
    user_id: UUID
    analysis_date: datetime
    readability: float
    sentiment_score: float
    self_talk_tone: str
    energy_score: float
    keywords: Dict[str, int]
    text_mood: Dict[str, float]
    emoji_mood: Dict[str, float]
    image_mood: Dict[str, float]
    combined_mood: Dict[str, float]
    goal_mentions: List[str]
    topics: List[Dict[str, str]]
    text_vector: str
    text_embedding: List[float]
    extracted_actions: str
    date: str
    model: str


class JournalAnalysisCreate(BaseSchema):
    journal_id: UUID
    readability: float
    sentiment_score: float
    self_talk_tone: str
    energy_score: float
    keywords: Dict[str, int]
    text_mood: Dict[str, float]
    emoji_mood: Dict[str, float]
    image_mood: Dict[str, float]
    combined_mood: Dict[str, float]
    goal_mentions: List[str]
    topics: List[Dict[str, str]]
    text_vector: str
    text_embedding: List[float]
    extracted_actions: str
    date: str
    model: str


class ConnectedAnalysisBase(BaseSchema):
    id: UUID
    user_id: UUID
    created_at: datetime
    mood_trends: Dict[str, Dict[str, float]]
    energy_trends: Dict[str, float]
    average_sentiment: float
    goal_emotion_map: Dict[str, Dict[str, float]]
    goal_progress: Dict[str, Dict]
    goal_matches: Dict[str, List[str]]
    keyword_emotion_map: Dict[str, Dict[str, float]]
    keyword_energy_map: Dict[str, float]
    journal_weights: Dict[str, float]
    model: str


class ConnectedAnalysisCreate(BaseSchema):
    mood_trends: Dict[str, Dict[str, float]]
    energy_trends: Dict[str, float]
    average_sentiment: float
    goal_emotion_map: Dict[str, Dict[str, float]]
    goal_progress: Dict[str, Dict]
    goal_matches: Dict[str, List[str]]
    keyword_emotion_map: Dict[str, Dict[str, float]]
    keyword_energy_map: Dict[str, float]
    journal_weights: Dict[str, float]
    model: str


class FeedbackBase(BaseSchema):
    id: UUID
    user_id: UUID
    connected_analysis_id: UUID
    tone: str
    feedback: str
    reflective_question: str
    motivation: str
    created_at: datetime


class FeedbackCreate(BaseSchema):
    connected_analysis_id: UUID
    tone: str
    feedback: str
    reflective_question: str
    motivation: str
