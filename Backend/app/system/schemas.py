from app.auth.schemas import UserOut
from pydantic import BaseModel

class DevLoginResponse(BaseModel):
    token: str
    user: UserOut

    class Config:
        from_attributes = True
