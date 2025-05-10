from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID

class BaseSchema(BaseModel):
    class Config:
        from_attributes = True

# User Schemas 
class UserBase(BaseSchema):
    email: str
    name: str
    type: str = "normal"
    auth_methods: List[str]
    storage_type: str
    storage_path: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: UUID

# Login Schemas 
class LoginRequest(BaseSchema):
    email: str
    password: str