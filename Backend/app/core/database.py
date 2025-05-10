from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
import app.auth.models
import app.goals.models
import app.journals.models

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
