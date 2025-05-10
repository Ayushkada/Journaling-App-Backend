from pydantic import BaseModel
from app.goals.schemas import GoalCreate
from app.journals.schemas import JournalEntryCreate
from typing import Optional, List


class BaseSchema(BaseModel):
    class Config:
        from_attributes = True

class AnalysisRequest(BaseSchema):
    journals: List[JournalEntryCreate]
    goals: Optional[List[GoalCreate]] = []