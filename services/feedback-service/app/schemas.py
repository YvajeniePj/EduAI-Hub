"""
Pydantic schemas for Feedback Service
"""
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


class FeedbackBase(BaseModel):
    user_name: str
    subject_id: Optional[UUID] = None
    group_id: Optional[UUID] = None
    quality_rating: int = Field(..., ge=1, le=5, description="Оценка качества обучения (1-5)")
    content_rating: int = Field(..., ge=1, le=5, description="Оценка содержания курса (1-5)")
    materials_rating: int = Field(..., ge=1, le=5, description="Оценка материалов (1-5)")
    support_rating: int = Field(..., ge=1, le=5, description="Оценка поддержки (1-5)")
    comment: Optional[str] = None
    suggestions: Optional[str] = None


class FeedbackCreate(FeedbackBase):
    pass


class FeedbackResponse(FeedbackBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackStats(BaseModel):
    subject_id: Optional[UUID] = None
    group_id: Optional[UUID] = None
    total_responses: int
    avg_quality_rating: float
    avg_content_rating: float
    avg_materials_rating: float
    avg_support_rating: float
    overall_avg: float

