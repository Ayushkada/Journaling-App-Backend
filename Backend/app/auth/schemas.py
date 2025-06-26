from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID


class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        orm_mode = True


class UserBase(BaseSchema):
    email: str
    name: str
    type: str = "normal"
    auth_methods: List[str]
    storage_type: str
    storage_path: Optional[str] = None


class UserCreate(UserBase):
    password: str  # Required only during sign-up via email/password


class UserOut(UserBase):
    id: UUID


class LoginRequest(BaseSchema):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    user: UserOut


class GoogleLoginRequest(BaseSchema):
    code: str
