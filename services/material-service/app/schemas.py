"""
Pydantic schemas for Material Service
"""
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Optional, List


class MaterialBase(BaseModel):
    subject_id: UUID
    name: str
    original_name: Optional[str] = None
    note: Optional[str] = None
    allowed_groups: Optional[List[str]] = None


class MaterialCreate(MaterialBase):
    pass


class MaterialResponse(MaterialBase):
    id: UUID
    path: str
    size: int
    mime_type: str
    uploader: str
    annotation_ru: Optional[str] = None
    annotation_en: Optional[str] = None
    annotation: Optional[str] = None  # Deprecated: kept for backward compatibility
    created_at: datetime

    class Config:
        from_attributes = True

