from sqlalchemy import Column, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from app.core.database import Base
import uuid

class Goal(Base):
    __tablename__ = "goals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    parent_goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=True)

    content = Column(String, nullable=False)
    ai_generated = Column(Boolean, default=False)
    category = Column(String, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
    emotion_trend = Column(JSON, nullable=True)
    first_mentioned_at = Column(DateTime, nullable=True)
    last_mentioned_at = Column(DateTime, nullable=True)
    notes = Column(String, nullable=True)
    progress_score = Column(Float, nullable=True)
    related_entry_ids = Column(ARRAY(String), nullable=True)
    time_limit = Column(DateTime, nullable=True)
    verified = Column(Boolean, default=False)
