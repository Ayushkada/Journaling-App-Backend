from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
from typing import Dict, Any

from app.core.database import get_db
from app.auth.models import User
from app.auth.schemas import UserCreate, LoginRequest, UserOut
from app.core.config import (
    APPLE_AUDIENCE,
    APPLE_ISSUER,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    APPLE_KEYS_URL,
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_token(user_id: UUID) -> str:
    payload = {
        "user_id": str(user_id),
        "exp": datetime.now(timezone.utc)
        + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
) -> UUID:
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user ID in token")
    try:
        return UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Malformed user ID in token")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user ID in token")
    user = db.query(User).filter(User.id == UUID(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def verify_apple_token(identity_token: str) -> dict:
    response = requests.get(APPLE_KEYS_URL)
    apple_keys = response.json()["keys"]
    header = jwt.get_unverified_header(identity_token)
    key = next(k for k in apple_keys if k["kid"] == header["kid"])
    public_key = jwt.construct_rsa_key(key)

    return jwt.decode(
        identity_token,
        public_key,
        algorithms=["RS256"],
        audience=APPLE_AUDIENCE,
        issuer=APPLE_ISSUER,
    )


def handle_login(user: LoginRequest, db: Session) -> Dict[str, Any]:
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="No account found with this email.")
    if "password" not in db_user.auth_methods:
        raise HTTPException(
            status_code=403,
            detail="This account is linked to Apple login. Please sign in using that method.",
        )
    if not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    token = create_token(db_user.id)
    return {
        "token": token,
        "user": UserOut.model_validate(db_user),  # Always return Pydantic object
    }


def handle_signup(user: UserCreate, db: Session) -> UserOut:
    if not user.password:
        raise HTTPException(
            status_code=400, detail="Password is required for email/password signup."
        )
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already exists.")

    new_user = User(
        id=uuid4(),
        email=user.email,
        name=user.name,
        password=hash_password(user.password),
        auth_methods=["password"],
        storage_type=user.storage_type,
        storage_path=user.storage_path,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserOut.model_validate(new_user)


def handle_apple_login(data: dict, db: Session) -> Dict[str, Any]:
    token = data.get("identityToken")
    if not token:
        raise HTTPException(status_code=400, detail="Missing Apple identity token.")
    try:
        payload = verify_apple_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=401, detail=f"Invalid or expired Apple token: {e}"
        )

    apple_id = payload["sub"]
    email = payload.get("email")
    user = (
        db.query(User)
        .filter((User.apple_sub == apple_id) | (User.email == email))
        .first()
    )

    if not user:
        user = User(
            id=uuid4(),
            email=email or f"{apple_id}@appleid.apple.com",
            name="Apple User",
            password=None,
            auth_methods=["apple"],
            apple_sub=apple_id,
            storage_type="local",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif "apple" not in user.auth_methods:
        if user.apple_sub and user.apple_sub != apple_id:
            raise HTTPException(
                status_code=403, detail="This email is already used by another account."
            )
        user.auth_methods.append("apple")
        user.apple_sub = apple_id
        db.commit()
        db.refresh(user)

    jwt_token = create_token(user.id)
    return {"token": jwt_token, "user": UserOut.model_validate(user)}
