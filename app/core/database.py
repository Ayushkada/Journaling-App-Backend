"""
Database configuration and session management for SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import DATABASE_URL

# Engine & Session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False, bind=engine
)

# Declarative Base
Base = declarative_base()

# Import all models to register them with the Base metadata
import app.auth.models  # noqa: F401
import app.goals.models  # noqa: F401
import app.journals.models  # noqa: F401
import app.analysis.models  # noqa: F401


# Dependency for FastAPI Routes
def get_db():
    """
    Yields a database session for use in FastAPI dependency injection.
    Ensures the session is closed after the request lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
