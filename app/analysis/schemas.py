# schemas.py
from typing import List, Dict, Optional, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class BaseSchema(BaseModel):
    class Config:
        from_attributes = True


from app.journals.schemas import JournalEntryCreate
from app.goals.schemas import GoalCreate


class AnalysisRequest(BaseSchema):
    journals: List[JournalEntryCreate]
    goals: Optional[List[GoalCreate]] = []


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

class PromptCatalogBase(BaseSchema):
    id: UUID
    text: str
    tone: Optional[str] = None
    tags: Optional[List[str]] = None
    time_estimate: Optional[float] = None
    source: Literal["base", "ai"]


class PromptCatalogCreate(BaseSchema):
    text: str
    tone: Optional[str] = None
    tags: Optional[List[str]] = None
    time_estimate: Optional[float] = None
    source: Literal["base", "ai"] = "base"


class UserPromptBase(BaseSchema):
    id: UUID
    user_id: UUID
    catalog_id: Optional[UUID] = None
    text: str
    tone: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Literal["base", "ai"]
    is_favorite: bool


class UserPromptCreate(BaseSchema):
    catalog_id: Optional[UUID] = None
    text: str
    tone: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Literal["base", "ai"] = "ai"
    is_favorite: bool = True


class PromptInteractionCreate(BaseSchema):
    catalog_id: Optional[UUID] = None
    prompt_text: str
    event: Literal["view", "start", "complete", "thumbs_up", "thumbs_down"]

class EntryLLMResponse(BaseModel):
    readability: float = 0.0
    sentimentScore: float = 0.0
    selfTalkTone: str = "NEUTRAL"
    energyScore: float = 0.0
    keywords: Dict[str, int] = {}
    textMood: Dict[str, float] = {}
    emojiMood: Dict[str, float] = {}
    imageMood: Dict[str, float] = {}
    mood: Dict[str, float] = {}
    goalMentions: List[str] = []
    topics: List[Dict[str, str]] = []
    textVector: Optional[str] = None
    extractedActions: str = ""


class ConnectedLLMResponse(BaseModel):
    moodTrends: Dict[str, Dict[str, float]] = {}
    energyTrends: Dict[str, float] = {}
    averageSentiment: float = 0.0
    goalEmotionMap: Dict[str, Dict[str, float]] = {}
    goalProgress: Dict[str, Dict] = {}
    goalMatches: Dict[str, List[str]] = {}
    keywordEmotionMap: Dict[str, Dict[str, float]] = {}
    keywordEnergyMap: Dict[str, float] = {}
    journalWeights: Dict[str, float] = {}
