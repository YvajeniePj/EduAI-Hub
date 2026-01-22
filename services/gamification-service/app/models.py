"""
Database models for Gamification Service
"""
from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.database import Base


class Points(Base):
    __tablename__ = "points"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user = Column(String, nullable=False)
    subject_id = Column(UUID(as_uuid=True), nullable=True)  # No foreign key in microservices
    points = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Points(id={self.id}, user={self.user}, points={self.points})>"

