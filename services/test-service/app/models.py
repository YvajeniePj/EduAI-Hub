"""
Database models for Test Service
"""
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Enum as SQLEnum, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class TestType(str, enum.Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    KEYWORD_BASED = "keyword_based"


class Test(Base):
    __tablename__ = "tests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), nullable=False)  # Foreign key removed for microservice independence
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    assignment_id = Column(String, nullable=True)  # For backward compatibility
    test_type = Column(SQLEnum(TestType), nullable=False)
    due_date = Column(DateTime, nullable=True)
    available_until = Column(DateTime, nullable=True)  # Дата до которой тест доступен для прохождения
    time_limit_minutes = Column(Integer, nullable=True)
    ai_generated = Column(String, default="false")  # Store as string for flexibility
    allowed_groups = Column(JSON, nullable=True)  # List of Group IDs allowed to view this test
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    questions = relationship("Question", back_populates="test", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Test(id={self.id}, title={self.title}, type={self.test_type})>"


class Question(Base):
    __tablename__ = "questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False)
    question_id = Column(String, nullable=False)  # q1, q2, etc.
    title = Column(String, nullable=False)
    max_points = Column(Integer, nullable=False)
    test_type = Column(SQLEnum(TestType), nullable=False)
    
    # For multiple_choice
    options = Column(JSON, nullable=True)  # List of strings
    correct_answer = Column(String, nullable=True)
    
    # Relationships
    test = relationship("Test", back_populates="questions")
    keywords = relationship("Keyword", back_populates="question", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Question(id={self.id}, question_id={self.question_id}, title={self.title})>"


class Keyword(Base):
    __tablename__ = "keywords"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id = Column(UUID(as_uuid=True), ForeignKey("questions.id"), nullable=False)
    word = Column(String, nullable=False)
    points = Column(Integer, nullable=False)
    
    # Relationships
    question = relationship("Question", back_populates="keywords")

    def __repr__(self):
        return f"<Keyword(id={self.id}, word={self.word}, points={self.points})>"

