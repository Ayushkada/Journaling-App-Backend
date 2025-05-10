from sqlalchemy import Column, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from app.core.database import Base

class Goal(Base):
    __tablename__ = "goals"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    content = Column(String, nullable=False)
    aiGenerated = Column(Boolean, default=False)
    category = Column(String, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
    emotion_trend = Column(JSON, nullable=True)
    first_mensioned_at = Column(DateTime, nullable=True)
    last_mensioned_at = Column(DateTime, nullable=True)
    notes = Column(String, nullable=True)
    progress_score = Column(Float, nullable=True)
    related_entry_ids = Column(ARRAY(String), nullable=True)
    time_limit = Column(DateTime, nullable=True)
    verified = Column(Boolean, default=False)
