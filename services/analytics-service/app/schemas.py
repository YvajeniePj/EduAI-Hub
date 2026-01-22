"""
Pydantic schemas for Analytics Service
"""
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Optional, List


class UserActivityBase(BaseModel):
    user_name: str
    action_type: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    session_duration: Optional[int] = None


class UserActivityCreate(UserActivityBase):
    pass


class UserActivityResponse(UserActivityBase):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


class StudentProgressBase(BaseModel):
    user_name: str
    subject_id: Optional[UUID] = None
    group_id: Optional[UUID] = None


class StudentProgressResponse(StudentProgressBase):
    id: UUID
    tests_completed: int
    tests_total: int
    average_score: Optional[float] = None
    total_time_seconds: int
    login_count: int
    materials_viewed: int
    videos_viewed: int
    last_activity_at: Optional[datetime] = None
    updated_at: datetime

    class Config:
        from_attributes = True


class AnalyticsReport(BaseModel):
    user_name: Optional[str] = None
    subject_id: Optional[UUID] = None
    group_id: Optional[UUID] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    total_students: int
    average_score: float
    total_time_hours: float
    total_tests_completed: int
    total_logins: int
    engagement_score: float  # Calculated metric (0-100)


class ActivityStats(BaseModel):
    user_name: str
    date: str  # YYYY-MM-DD
    total_time_seconds: int
    login_count: int
    test_actions: int
    material_views: int
    video_views: int

