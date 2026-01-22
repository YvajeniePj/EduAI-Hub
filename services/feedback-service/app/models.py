"""
Database models for Feedback Service
"""
from sqlalchemy import Column, String, DateTime, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

from app.database import Base


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_name = Column(String, nullable=False)  # Store username as string
    subject_id = Column(UUID(as_uuid=True), nullable=True)  # Optional: feedback about specific subject
    group_id = Column(UUID(as_uuid=True), nullable=True)  # Optional: feedback about specific group
    
    # Feedback questions (1-5 scale)
    quality_rating = Column(Integer, nullable=False)  # Оценка качества обучения (1-5)
    content_rating = Column(Integer, nullable=False)  # Оценка содержания курса (1-5)
    materials_rating = Column(Integer, nullable=False)  # Оценка материалов (1-5)
    support_rating = Column(Integer, nullable=False)  # Оценка поддержки (1-5)
    
    # Additional feedback
    comment = Column(Text, nullable=True)  # Дополнительные комментарии
    suggestions = Column(Text, nullable=True)  # Предложения по улучшению
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Feedback(id={self.id}, user_name={self.user_name}, quality_rating={self.quality_rating})>"

