"""
Database models for Submission Service
"""
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_id = Column(UUID(as_uuid=True), nullable=False)  # Reference to test
    user = Column(String, nullable=False)
    assignment = Column(String, nullable=True)  # For backward compatibility
    total_score = Column(Integer, default=0)
    total_max = Column(Integer, default=0)
    points_awarded = Column(Integer, default=0)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    is_finished = Column(String, default="false")  # Store as string for flexibility

    # Relationships
    answers = relationship("Answer", back_populates="submission", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Submission(id={self.id}, user={self.user}, test_id={self.test_id})>"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    avatar_url = Column(String, nullable=True)
    role = Column(String, default="student")

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name})>"


class Answer(Base):
    __tablename__ = "answers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False)
    question_id = Column(String, nullable=False)  # q1, q2, etc.
    answer = Column(String, nullable=False)
    score = Column(Integer, default=0)
    ai_score = Column(Integer, nullable=True)
    final_score = Column(Integer, nullable=True)
    ai_feedback = Column(JSON, nullable=True)  # List of strings
    details = Column(JSON, nullable=True)  # List of strings

    # Relationships
    submission = relationship("Submission", back_populates="answers")

    def __repr__(self):
        return f"<Answer(id={self.id}, question_id={self.question_id}, score={self.score})>"

