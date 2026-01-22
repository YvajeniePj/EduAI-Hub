"""
Database models for Analytics Service
"""
from sqlalchemy import Column, String, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.database import Base


class UserActivity(Base):
    __tablename__ = "user_activities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_name = Column(String, nullable=False)  # Store username as string
    action_type = Column(String, nullable=False)  # login, test_start, test_finish, material_view, video_view, etc.
    resource_type = Column(String, nullable=True)  # test, material, video, subject, etc.
    resource_id = Column(String, nullable=True)  # ID of the resource
    session_duration = Column(Integer, nullable=True)  # Duration in seconds
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<UserActivity(id={self.id}, user_name={self.user_name}, action_type={self.action_type})>"


class StudentProgress(Base):
    __tablename__ = "student_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_name = Column(String, nullable=False)
    subject_id = Column(UUID(as_uuid=True), nullable=True)
    group_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Progress metrics
    tests_completed = Column(Integer, default=0, nullable=False)
    tests_total = Column(Integer, default=0, nullable=False)
    average_score = Column(Float, nullable=True)
    
    # Activity metrics (calculated from user_activities)
    total_time_seconds = Column(Integer, default=0, nullable=False)  # Total time in system
    login_count = Column(Integer, default=0, nullable=False)
    materials_viewed = Column(Integer, default=0, nullable=False)
    videos_viewed = Column(Integer, default=0, nullable=False)
    
    # Last updated
    last_activity_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<StudentProgress(id={self.id}, user_name={self.user_name}, average_score={self.average_score})>"

