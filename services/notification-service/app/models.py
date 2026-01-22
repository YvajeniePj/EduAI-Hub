"""
Database models for Notification Service
"""
from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.database import Base


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_name = Column(String, nullable=False)  # Store username as string
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String, nullable=False, default="info")  # info, warning, error, success
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    read_at = Column(DateTime, nullable=True)
    
    # Optional fields for linking to specific resources
    related_type = Column(String, nullable=True)  # test, submission, subject, etc.
    related_id = Column(String, nullable=True)  # ID of related resource

    def __repr__(self):
        return f"<Notification(id={self.id}, user_name={self.user_name}, title={self.title})>"

