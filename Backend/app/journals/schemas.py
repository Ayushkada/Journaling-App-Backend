from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        orm_mode = True


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


class JournalEntryUpdate(BaseSchema):
    title: Optional[str] = None
    content: Optional[str] = None
    date: Optional[datetime] = None
    emojis: Optional[List[str]] = None
    images: Optional[List[str]] = None
    analyze_images: Optional[bool] = None
    source: Optional[str] = None
    analysis_status: Optional[str] = None
