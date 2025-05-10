from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.auth.models import User
from app.auth.schemas import UserCreate, UserOut, LoginRequest
from app.auth.service import (
    handle_login,
    handle_signup,
    handle_apple_login,
    get_current_user,
)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login", response_model=UserOut)
def login(user: LoginRequest, db: Session = Depends(get_db)) -> UserOut:
    return handle_login(user, db)


@router.post("/signup", response_model=UserOut)
def signup(user: UserCreate = Body(...), db: Session = Depends(get_db)) -> UserOut:
    return handle_signup(user, db)


@router.post("/apple-login", response_model=UserOut)
def apple_login(data: dict = Body(...), db: Session = Depends(get_db)) -> UserOut:
    return handle_apple_login(data, db)


@router.get("/me", response_model=UserOut)
def get_profile(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(user)
