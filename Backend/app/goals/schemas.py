from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field

class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        orm_mode = True

class GoalBase(BaseSchema):
    id: UUID
    user_id: UUID
    parent_goal_id: Optional[UUID] = None
    content: str
    ai_generated: bool
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
    verified: bool = False

class GoalCreate(BaseSchema):
    content: str
    ai_generated: bool
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
    parent_goal_id: Optional[UUID] = None

class GoalUpdate(BaseSchema):
    content: Optional[str] = None
    ai_generated: Optional[bool] = None
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
    parent_goal_id: Optional[UUID] = None

class GoalResponse(GoalBase):
    pass
