"""
Pydantic schemas for Test Service
"""
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone, timedelta
from uuid import UUID
from typing import List, Optional
from app.models import TestType


# Keyword schemas
class KeywordBase(BaseModel):
    word: str
    points: int


class KeywordCreate(KeywordBase):
    pass


class KeywordResponse(KeywordBase):
    id: UUID

    class Config:
        from_attributes = True


# Question schemas
class QuestionBase(BaseModel):
    question_id: str
    title: str
    max_points: int
    test_type: TestType


class QuestionCreate(QuestionBase):
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    keywords: Optional[List[KeywordCreate]] = None


class QuestionResponse(QuestionBase):
    id: UUID
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    keywords: List[KeywordResponse] = []

    class Config:
        from_attributes = True


# Test schemas
class TestBase(BaseModel):
    subject_id: UUID
    title: str
    description: Optional[str] = None
    assignment_id: Optional[str] = None
    test_type: TestType
    due_date: Optional[datetime] = None
    available_until: Optional[datetime] = None  # Дата до которой тест доступен для прохождения
    time_limit_minutes: Optional[int] = None
    time_limit_minutes: Optional[int] = None
    ai_generated: Optional[bool] = False
    allowed_groups: Optional[List[str]] = None


class TestCreate(TestBase):
    questions: Optional[List[QuestionCreate]] = []


class TestUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    available_until: Optional[datetime] = None
    available_until: Optional[datetime] = None
    time_limit_minutes: Optional[int] = None
    allowed_groups: Optional[List[str]] = None


class TestResponse(TestBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    questions: List[QuestionResponse] = []

    @field_validator('due_date', 'available_until', mode='before')
    @classmethod
    def add_moscow_timezone(cls, v):
        """Добавляем московский часовой пояс (+03:00) к датам при возврате"""
        if v is None:
            return v
        # Если дата уже имеет timezone, конвертируем в московское время
        if isinstance(v, datetime):
            moscow_tz = timezone(timedelta(hours=3))
            if v.tzinfo is None:
                # Если timezone нет (UTC из базы), предполагаем что это UTC и конвертируем в московское
                return v.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
            else:
                # Если timezone есть, конвертируем в московское
                return v.astimezone(moscow_tz)
        return v

    class Config:
        from_attributes = True

