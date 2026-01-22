"""
Pydantic schemas for Video Service
"""
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Optional, Dict, List


class VideoBase(BaseModel):
    subject_id: UUID
    url: str
    title: str
    note: Optional[str] = None
    uploader: str
    video_info: Optional[Dict] = None
    allowed_groups: Optional[List[str]] = None


class VideoCreate(VideoBase):
    pass


class VideoResponse(VideoBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

