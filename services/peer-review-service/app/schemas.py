"""
Pydantic schemas for Peer Review Service
"""
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


class ReviewScores(BaseModel):
    relevance: int = Field(..., ge=1, le=5)
    structure: int = Field(..., ge=1, le=5)
    argument: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)


class ReviewBase(BaseModel):
    submission_id: UUID
    assignment_id: Optional[str] = None
    reviewer: str
    relevance: int = Field(..., ge=1, le=5)
    structure: int = Field(..., ge=1, le=5)
    argument: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class ReviewCreate(ReviewBase):
    pass


class ReviewResponse(ReviewBase):
    id: UUID
    avg_score: float
    created_at: datetime

    class Config:
        from_attributes = True

