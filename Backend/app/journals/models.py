import datetime
from sqlalchemy import Column, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from app.core.database import Base


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    title = Column(String, nullable=True)
    content = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    emojis = Column(ARRAY(String), nullable=True)
    images = Column(ARRAY(String), nullable=True)
    analyze_images = Column(Boolean, default=False)
    source = Column(String, nullable=False)
    
    analysis_status = Column(String, nullable=False, default="pending") 

class JournalAnalysis(Base):
    __tablename__ = "journal_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True)
    journal_id = Column(UUID(as_uuid=True), nullable=False)
    analysis_date = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    readability = Column(Float)
    sentiment_score = Column(Float)
    self_talk_tone = Column(String)
    energy_score = Column(Float)
    keywords = Column(JSON)
    text_mood = Column(JSON)
    emoji_mood = Column(JSON)
    image_mood = Column(JSON)
    combined_mood = Column(JSON)
    goal_mentions = Column(JSON)
    topics = Column(JSON)
    text_vector = Column(String)
    text_embedding = Column(JSON)
    extracted_actions = Column(String)
    date = Column(String)
    model = Column(String, nullable=False)

class ConnectedAnalysis(Base):
    __tablename__ = "connected_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    mood_trends = Column(JSON)
    energy_trends = Column(JSON)
    average_sentiment = Column(Float)
    goal_emotion_map = Column(JSON)
    goal_progress = Column(JSON)
    goal_matches = Column(JSON)
    keyword_emotion_map = Column(JSON)
    keyword_energy_map = Column(JSON)
    journal_weights = Column(JSON)
    model = Column(String, nullable=False)