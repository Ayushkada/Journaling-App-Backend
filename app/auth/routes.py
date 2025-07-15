from typing import Optional
import logging

from fastapi import APIRouter, Depends, Body, HTTPException, Cookie, Response, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.models import User
from app.auth.schemas import (
    ChangeEmailRequest,
    ChangePasswordRequest,
    ChangeUsernameRequest,
    UserCreate,
    UserOut,
    LoginRequest,
    GoogleLoginRequest,
    TokenResponse,
)
from app.auth.service import (
    handle_login,
    handle_signup,
    handle_apple_login,
    handle_google_login,
    get_current_user,
    handle_token_refresh,
    update_user_email,
    update_user_password,
    update_user_username,
)
from app.core.config import GOOGLE_CLIENT_ID, GOOGLE_REDIRECT_URI

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])


def set_refresh_cookie(response: Response, refresh_token: str):
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive access/refresh tokens",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        500: {"description": "Server error"},
    },
)
def login_route(
    user: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> TokenResponse:
    try:
        token_response, refresh_token = handle_login(user, db)
        set_refresh_cookie(response, refresh_token)
        return token_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post(
    "/signup",
    response_model=UserOut,
    summary="Register a new user",
    responses={
        200: {"description": "User created successfully"},
        400: {"description": "User already exists or validation error"},
        500: {"description": "Signup failed"},
    },
)
def signup_route(
    user: UserCreate = Body(...),
    response: Response = None,
    db: Session = Depends(get_db),
) -> UserOut:
    try:
        token_response, refresh_token = handle_signup(user, db)
        set_refresh_cookie(response, refresh_token)
        return token_response.user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        raise HTTPException(status_code=500, detail="Signup failed")


@router.post(
    "/apple-login",
    response_model=TokenResponse,
    summary="Login using Apple credentials",
    responses={
        200: {"description": "Apple login successful"},
        500: {"description": "Apple login failed"},
    },
)
def apple_login_route(
    data: dict = Body(...),
    db: Session = Depends(get_db),
) -> TokenResponse:
    try:
        return handle_apple_login(data, db)
    except Exception as e:
        logger.error(f"Apple login failed: {e}")
        raise HTTPException(status_code=500, detail="Apple login failed")


@router.post(
    "/google-login",
    response_model=TokenResponse,
    summary="Login using Google credentials",
    responses={
        200: {"description": "Google login successful"},
        500: {"description": "Google login failed"},
    },
)
def google_login_route(
    data: GoogleLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    try:
        return handle_google_login(data, db)
    except Exception as e:
        logger.error(f"Google login failed: {e}")
        raise HTTPException(status_code=500, detail="Google login failed")


@router.get(
    "/google/url",
    summary="Get Google OAuth URL",
    responses={
        200: {"description": "OAuth URL returned"},
        500: {"description": "Missing Google OAuth config"},
    },
)
def get_google_oauth_url_route() -> dict:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    oauth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={GOOGLE_REDIRECT_URI}&"
        "response_type=code&"
        "scope=openid%20email%20profile&"
        "access_type=offline"
    )
    return {"url": oauth_url}


@router.get(
    "/me",
    response_model=UserOut,
    summary="Get current user profile",
    responses={
        200: {"description": "User profile returned"},
        401: {"description": "Unauthorized"},
    },
)
def get_profile_route(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token using refresh token",
    responses={
        200: {"description": "Access token refreshed"},
        401: {"description": "No or invalid refresh token"},
        500: {"description": "Token refresh failed"},
    },
)
def refresh_token_route(
    refresh_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db),
    response: Response = None,
) -> TokenResponse:
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token found")

    try:
        token_response, new_refresh_token = handle_token_refresh(refresh_token, db)
        set_refresh_cookie(response, new_refresh_token)
        return token_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


@router.post(
    "/logout",
    summary="Log out the user and clear refresh token cookie",
    responses={
        200: {"description": "User logged out successfully"},
    },
)
def logout_route(response: Response) -> dict:
    response.delete_cookie(
        key="refresh_token",
        path="/",  # should match the path used when setting the cookie
        samesite="none",
        secure=True,
        httponly=True,
    )
    return {"detail": "Logged out"}


@router.put(
    "/change-email",
    response_model=UserOut,
    summary="Change user email",
)
def change_email(
    request: ChangeEmailRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserOut:
    updated_user = update_user_email(user, request.new_email, db)
    return UserOut.model_validate(updated_user)


@router.put(
    "/change-username",
    response_model=UserOut,
    summary="Change user username",
)
def change_username(
    request: ChangeUsernameRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserOut:
    updated_user = update_user_username(user, request.new_username, db)
    return UserOut.model_validate(updated_user)


@router.put(
    "/change-password",
    summary="Change user password",
    responses={
        200: {"description": "Password updated"},
        400: {"description": "Old password is incorrect"},
    },
)
def change_password(
    request: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    update_user_password(user, request.old_password, request.new_password, db)
    return {"detail": "Password updated successfully"}


@router.delete("/delete", summary="Delete account")
def delete_account(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    db.delete(user)
    db.commit()
    return {"detail": "Account deleted successfully"}
