from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class RoomCreate(BaseModel):
    subject_id: UUID
    teacher_name: str
    room_name: Optional[str] = None

class JoinRequest(BaseModel):
    room_name: str
    identity: str
    is_teacher: bool = False

class TokenResponse(BaseModel):
    token: str
    room_name: str
    server_url: str

class RoomResponse(BaseModel):
    id: UUID
    subject_id: UUID
    room_name: str
    teacher_name: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
