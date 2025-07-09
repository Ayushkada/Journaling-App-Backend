import logging
from fastapi import HTTPException, Depends, Security
from sqlalchemy.orm import Session
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional, Tuple
import requests

from app.core.database import get_db
from app.auth.models import User
from app.auth.schemas import (
    UserCreate, LoginRequest, UserOut,
    GoogleLoginRequest, TokenResponse
)
from app.core.config import (
    APPLE_AUDIENCE, APPLE_ISSUER, APPLE_KEYS_URL,
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI,
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
)

# Initialize logger and security tools
logger = logging.getLogger(__name__)
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer()


def hash_password(password: str) -> str:
    """
    Hashes a plaintext password using bcrypt.

    Args:
        password (str): Raw password input.

    Returns:
        str: Bcrypt-hashed password.
    """
    return pwd.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a bcrypt hash.

    Args:
        plain_password (str): Input password to check.
        hashed_password (str): Stored hashed password.

    Returns:
        bool: True if match, False otherwise.
    """
    return pwd.verify(plain_password, hashed_password)


def create_token(user_id: UUID, token_type: str = "access", expires_delta: Optional[timedelta] = None) -> str:
    now = datetime.now(timezone.utc)
    if not expires_delta:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) if token_type == "access" else timedelta(days=7)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now.timestamp(),
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decodes and validates a JWT token.

    Args:
        token (str): JWT string.

    Returns:
        dict: Decoded payload.
    
    Raises:
        HTTPException: If token is invalid or expired.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_aud": False})
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")


def get_current_user_id(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
) -> UUID:
    """
    Extracts the user ID from the JWT token.

    Args:
        creds (HTTPAuthorizationCredentials): Bearer token.

    Returns:
        UUID: User's UUID.

    Raises:
        HTTPException: If token is invalid or missing required claims.
    """
    payload = decode_token(creds.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject field")
    try:
        return UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user ID in token")


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    Fetches the full user object from the database using token credentials.

    Args:
        creds (HTTPAuthorizationCredentials): Bearer token.
        db (Session): SQLAlchemy session.

    Returns:
        User: Authenticated user instance.

    Raises:
        HTTPException: If user not found.
    """
    user_id = get_current_user_id(creds)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def verify_apple_token(identity_token: str) -> dict:
    """
    Verifies and decodes an Apple identity token.

    Args:
        identity_token (str): Apple JWT token.

    Returns:
        dict: Decoded user identity data.

    Raises:
        HTTPException: If the token is invalid or cannot be verified.
    """
    resp = requests.get(APPLE_KEYS_URL)
    resp.raise_for_status()
    keys = resp.json().get("keys", [])
    header = jwt.get_unverified_header(identity_token)
    key = next((k for k in keys if k["kid"] == header["kid"]), None)
    if not key:
        raise HTTPException(status_code=401, detail="Invalid Apple token: key not found")
    public_key = jwt.construct_rsa_key(key)
    return jwt.decode(identity_token, public_key, algorithms=["RS256"], audience=APPLE_AUDIENCE, issuer=APPLE_ISSUER)


def exchange_google_code_for_tokens(code: str) -> Dict[str, Any]:
    """
    Exchanges Google OAuth code for access and ID tokens.

    Args:
        code (str): Authorization code.

    Returns:
        dict: Token payload from Google.

    Raises:
        HTTPException: If the exchange fails.
    """
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": GOOGLE_REDIRECT_URI,
        }
    )
    if resp.status_code != 200:
        logger.error("Google token exchange failed: %s", resp.text)
        raise HTTPException(status_code=400, detail="Invalid Google auth code")
    return resp.json()


def get_google_user_info(access_token: str) -> Dict[str, Any]:
    """
    Retrieves Google user profile using the access token.

    Args:
        access_token (str): Google OAuth token.

    Returns:
        dict: Google user profile data.

    Raises:
        HTTPException: If retrieval fails.
    """
    resp = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    if resp.status_code != 200:
        logger.error("Failed to fetch Google user info: %s", resp.text)
        raise HTTPException(status_code=400, detail="Failed to fetch Google user info")
    return resp.json()


def handle_google_login(data: GoogleLoginRequest, db: Session) -> TokenResponse:
    """
    Logs in or registers a user using Google OAuth.

    Args:
        data (GoogleLoginRequest): Auth code request from frontend.
        db (Session): DB session.

    Returns:
        TokenResponse: JWT token and user data.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        logger.error("Google OAuth misconfigured")
        raise HTTPException(status_code=500, detail="OAuth not available")

    tokens = exchange_google_code_for_tokens(data.code)
    info = get_google_user_info(tokens["access_token"])
    email = info.get("email", "").lower().strip()
    if not email:
        raise HTTPException(status_code=400, detail="Google account missing email")

    user = db.query(User).filter((User.google_id == info["id"]) | (User.email == email)).first()
    created = False

    if not user:
        user = User(
            id=uuid4(), email=email, name=info.get("name", ""), 
            auth_methods=["google"], google_id=info["id"]
        )
        db.add(user); db.commit(); db.refresh(user)
        created = True
    elif "google" not in user.auth_methods:
        user.auth_methods.append("google"); user.google_id = info["id"]
        db.commit(); db.refresh(user)

    jwt_token = create_token(user.id)
    logger.info("Google login for user %s (new=%s)", user.id, created)
    return TokenResponse(access_token=jwt_token, user=UserOut.model_validate(user))


def handle_apple_login(data: Dict[str, Any], db: Session) -> TokenResponse:
    """
    Logs in or registers a user using Apple Sign-In.

    Args:
        data (dict): Payload containing the Apple identity token.
        db (Session): DB session.

    Returns:
        TokenResponse: JWT token and user data.
    """
    token = data.get("identityToken")
    if not token:
        raise HTTPException(status_code=400, detail="Missing Apple identity token")

    info = verify_apple_token(token)
    email = info.get("email", "").lower().strip()
    apple_id = info["sub"]

    user = db.query(User).filter((User.apple_sub == apple_id) | (User.email == email)).first()
    created = False

    if not user:
        user = User(
            id=uuid4(), email=email or f"{apple_id}@appleid.apple.com", name="Apple User",
            auth_methods=["apple"], apple_sub=apple_id
        )
        db.add(user); db.commit(); db.refresh(user)
        created = True
    elif "apple" not in user.auth_methods:
        user.auth_methods.append("apple"); user.apple_sub = apple_id
        db.commit(); db.refresh(user)

    jwt_token = create_token(user.id)
    logger.info("Apple login for user %s (new=%s)", user.id, created)
    return TokenResponse(access_token=jwt_token, user=UserOut.model_validate(user))


def handle_login(req: LoginRequest, db: Session) -> TokenResponse:
    """
    Handles login via email and password.

    Args:
        req (LoginRequest): Email and password credentials.
        db (Session): DB session.

    Returns:
        TokenResponse: JWT token and user object.
    """
    email = req.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user or "password" not in user.auth_methods:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_token(user.id, token_type="access")
    refresh_token = create_token(user.id, token_type="refresh")

    return TokenResponse(access_token=access_token, user=UserOut.model_validate(user)), refresh_token


def handle_signup(req: UserCreate, db: Session) -> Tuple[TokenResponse, str]:
    """
    Handles user signup using email and password.

    Args:
        req (UserCreate): Signup request data.
        db (Session): DB session.

    Returns:
        UserOut: Created user object.
    """
    email = req.email.lower().strip()
    if not req.password:
        raise HTTPException(status_code=400, detail="Password required")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = User(
        id=uuid4(), email=email, name=req.name,
        password=hash_password(req.password), auth_methods=["password"]
    )
    db.add(user); db.commit(); db.refresh(user)

    access_token = create_token(user.id, token_type="access")
    refresh_token = create_token(user.id, token_type="refresh")

    return TokenResponse(access_token=access_token, user=UserOut.model_validate(user)), refresh_token


def handle_token_refresh(refresh_token: str, db: Session) -> TokenResponse:
    """
    Verifies the refresh token and issues a new access + refresh token pair.

    Args:
        refresh_token (str): The token from the cookie.
        db (Session): Active DB session.

    Returns:
        TokenResponse: New access token and user info.
    """
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access_token = create_token(user.id, token_type="access")
    refresh_token = create_token(user.id, token_type="refresh")

    return TokenResponse(access_token=access_token, user=UserOut.model_validate(user)), refresh_token


def update_user_email(user: User, new_email: str, db: Session):
    """
    Updates the user's email address.

    Args:
        user (User): The currently authenticated user object.
        new_email (str): The new email address to update to.
        db (Session): Active DB session.

    Returns:
        User: The updated user object.
    """

    user.email = new_email

    db.commit()

    db.refresh(user)

    return user


def update_user_username(user: User, new_username: str, db: Session):
    """
    Updates the user's username.

    Args:
        user (User): The currently authenticated user object.
        new_username (str): The new username to update to.
        db (Session): Active DB session.

    Returns:
        User: The updated user object.
    """

    user.name = new_username
    db.commit()
    db.refresh(user)
    return user


def update_user_password(user: User, old_password: str, new_password: str, db: Session):
    """
    Updates the user's password after verifying the old password.

    Args:
        user (User): The currently authenticated user object.
        old_password (str): The user's current password (for verification).
        new_password (str): The new password to set.
        db (Session): Active DB session.

    Returns:
        User: The updated user object.

    Raises:
        HTTPException: If the old password does not match.
    """

    if not verify_password(old_password, user.password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")

    user.password = hash_password(new_password)

    db.commit()

    db.refresh(user)

    return user
