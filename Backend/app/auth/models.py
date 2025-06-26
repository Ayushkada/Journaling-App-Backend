from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password = Column(String, nullable=True)  # Null if OAuth only
    auth_methods = Column(ARRAY(String), default=["password"])
    apple_sub = Column(String, unique=True, nullable=True)
    google_id = Column(String, unique=True, nullable=True)
    type = Column(String, nullable=False, default="normal")  # "normal", "admin", etc.
    storage_type = Column(String, default="local")  # "local" or "icloud"
    storage_path = Column(String, nullable=True)

    # Optional: if using relationships in journals/goals
    journals = relationship("JournalEntry", back_populates="user", cascade="all, delete-orphan")
    goals = relationship("Goal", back_populates="user", cascade="all, delete-orphan")
