"""
Database models for Peer Review Service
"""
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.database import Base


class Review(Base):
    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), nullable=False)  # Reference to submission
    assignment_id = Column(String, nullable=True)
    reviewer = Column(String, nullable=False)
    relevance = Column(Integer, nullable=False)  # 1-5
    structure = Column(Integer, nullable=False)  # 1-5
    argument = Column(Integer, nullable=False)  # 1-5
    clarity = Column(Integer, nullable=False)  # 1-5
    avg_score = Column(Float, nullable=False)
    comment = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Review(id={self.id}, reviewer={self.reviewer}, avg_score={self.avg_score})>"

