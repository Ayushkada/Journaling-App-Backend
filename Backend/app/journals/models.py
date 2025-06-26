import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PG_UUID
from app.core.database import Base


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)

    title = Column(String, nullable=True)
    content = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)

    emojis = Column(ARRAY(String), nullable=True)
    images = Column(ARRAY(String), nullable=True)
    analyze_images = Column(Boolean, default=False)

    source = Column(String, nullable=False)  # e.g., "web", "mobile"
    analysis_status = Column(String, nullable=False, default="pending")  # pending, complete, failed
