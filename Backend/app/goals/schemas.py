from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID

class BaseSchema(BaseModel):
    class Config:
        from_attributes = True


class GoalBase(BaseSchema):
    id: UUID
    user_id: UUID
    content: str
    aiGenerated: bool
    category: Optional[str] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    emotion_trend: Optional[List[float]] = None
    first_mentioned_at: Optional[datetime] = None
    last_mentioned_at: Optional[datetime] = None
    notes: Optional[str] = None
    progress_score: float
    related_entry_ids: Optional[List[str]] = None  
    time_limit: Optional[datetime] = None
    verified: Optional[bool] = False

class GoalCreate(BaseSchema):
    content: str
    aiGenerated: bool
    category: Optional[str] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    emotion_trend: Optional[List[float]] = None
    first_mentioned_at: Optional[datetime] = None
    last_mentioned_at: Optional[datetime] = None
    notes: Optional[str] = None
    progress_score: float
    related_entry_ids: Optional[List[str]] = None
    time_limit: Optional[datetime] = None
    verified: Optional[bool] = False

class GoalResponse(GoalBase):
    pass

class GoalUpdate(BaseSchema):
    content: Optional[str] = None
    aiGenerated: Optional[bool] = None
    category: Optional[str] = None
    completed_at: Optional[datetime] = None
    emotion_trend: Optional[List[float]] = None
    first_mentioned_at: Optional[datetime] = None
    last_mentioned_at: Optional[datetime] = None
    notes: Optional[str] = None
    progress_score: Optional[float] = None
    related_entry_ids: Optional[List[str]] = None
    time_limit: Optional[datetime] = None
    verified: Optional[bool] = None