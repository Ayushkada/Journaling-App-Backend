from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from app.core.database import Base
import uuid


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password = Column(String, nullable=True)
    auth_methods = Column(ARRAY(String), default=["password"])
    apple_sub = Column(String, unique=True, nullable=True)
    type = Column(String, nullable=False, default="normal")
    storage_type = Column(String, default="local")
    storage_path = Column(String, nullable=True)
