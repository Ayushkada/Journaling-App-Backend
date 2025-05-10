from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.auth.models import User
from app.journals.models import JournalEntry
from app.goals.models import Goal as Goal
from app.auth.service import create_token
from app.auth.schemas import UserOut
from app.journals.schemas import JournalEntryResponse
from app.goals.schemas import GoalResponse
from app.system.schemas import DevLoginResponse
from pydantic import BaseModel
from uuid import UUID

router = APIRouter(prefix="/system", tags=["System"])


@router.post("/test-login/", response_model=DevLoginResponse)
def dev_login_route(db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == "test@gmail.com").first()
    if not user:
        raise HTTPException(status_code=404, detail="Test user not found")

    token = create_token(user.id)

    return DevLoginResponse(
        token=token,
        user=UserOut(
            id=user.id,
            email=user.email,
            name=user.name,
            auth_methods=user.auth_methods,
            storage_type=user.storage_type,
            storage_path=user.storage_path,
        ),
    )


@router.get("/debug/users", response_model=List[UserOut])
def get_users_route(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.email).all()
    return users


@router.get("/debug/journals", response_model=List[JournalEntryResponse])
def get_all_journals_route(db: Session = Depends(get_db)):
    journals = db.query(JournalEntry).order_by(JournalEntry.date.desc()).all()
    return [JournalEntryResponse.model_validate(journal) for journal in journals]


@router.get("/debug/goals", response_model=List[GoalResponse])
def get_all_goals_route(db: Session = Depends(get_db)):
    goals = db.query(Goal).order_by(Goal.created_at.desc()).all()
    return [GoalResponse.model_validate(goal) for goal in goals]
