"""
Database models for Video Service
"""
from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.database import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), nullable=False)  # No foreign key in microservices
    url = Column(String, nullable=False)
    title = Column(String, nullable=False)
    note = Column(String, nullable=True)
    uploader = Column(String, nullable=False)
    video_info = Column(JSON, nullable=True)  # Store video metadata
    allowed_groups = Column(JSON, nullable=True)  # List of Group IDs allowed to view this video
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Video(id={self.id}, title={self.title})>"

