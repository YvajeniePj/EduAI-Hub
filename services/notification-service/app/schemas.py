"""
Pydantic schemas for Notification Service
"""
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Optional


class NotificationBase(BaseModel):
    user_name: Optional[str] = None
    title: str
    message: str
    type: str = "info"
    related_type: Optional[str] = None
    related_id: Optional[str] = None
    exclude_user_name: Optional[str] = None


class NotificationCreate(NotificationBase):
    pass


class NotificationUpdate(BaseModel):
    is_read: Optional[bool] = None


class NotificationResponse(NotificationBase):
    id: UUID
    is_read: bool
    created_at: datetime
    read_at: Optional[datetime] = None

    class Config:
        from_attributes = True

