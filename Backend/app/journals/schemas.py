from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
from uuid import UUID


class BaseSchema(BaseModel):
    class Config:
        from_attributes = True

# Journal Entry Schemas
class JournalEntryBase(BaseSchema):
    id: UUID
    user_id: UUID
    title: Optional[str] = None
    content: str
    date: datetime
    emojis: Optional[List[str]] = None
    images: Optional[List[str]] = None
    analyze_images: bool = False
    source: str
    analysis_status: Optional[str] = "pending"

class JournalEntryCreate(BaseSchema):
    title: Optional[str] = None
    content: str
    date: datetime
    emojis: Optional[List[str]] = None
    images: Optional[List[str]] = None
    analyze_images: bool = False
    source: str

class JournalEntryResponse(JournalEntryBase):
    pass

class JournalEntryUpdate(BaseSchema):
    title: Optional[str] = None
    content: Optional[str] = None
    date: Optional[datetime] = None
    emojis: Optional[List[str]] = None
    images: Optional[List[str]] = None
    analyze_images: Optional[bool] = None
    source: Optional[str] = None
    analysis_status: Optional[str] = None

# Journal Analysis Schemas
class JournalAnalysisBase(BaseSchema):
    id: UUID
    journal_id: UUID
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
    topics: List[Dict]
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
    topics: List[Dict]
    text_vector: str
    text_embedding: List[float]
    extracted_actions: str
    date: str
    model: str

class JournalAnalysisResponse(JournalAnalysisBase):
    pass

# Connected Analysis Schemas
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
    user_id: UUID
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

class ConnectedAnalysisResponse(ConnectedAnalysisBase):
    pass
