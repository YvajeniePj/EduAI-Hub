"""
Database models for Subject Service
"""
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


class Subject(Base):
    __tablename__ = "subjects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    cover_image = Column(String, nullable=True)  # URL or path to cover image
    rating = Column(Float, nullable=True)  # Average rating (0-5)
    rating_count = Column(Integer, default=0, nullable=False)  # Number of ratings
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    news = relationship("News", back_populates="subject", cascade="all, delete-orphan")
    groups = relationship("Group", back_populates="subject", cascade="all, delete-orphan")
    modules = relationship("CourseModule", back_populates="subject", cascade="all, delete-orphan", order_by="CourseModule.order_index")

    def __repr__(self):
        return f"<Subject(id={self.id}, name={self.name})>"


class News(Base):
    __tablename__ = "news"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), ForeignKey("subjects.id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    subject = relationship("Subject", back_populates="news")

    def __repr__(self):
        return f"<News(id={self.id}, title={self.title}, subject_id={self.subject_id})>"


class Group(Base):
    __tablename__ = "groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), ForeignKey("subjects.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    max_size = Column(Integer, nullable=True)  # Max number of members
    created_by = Column(String, nullable=True)  # Username of creator
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    subject = relationship("Subject", back_populates="groups")
    members = relationship("GroupMember", back_populates="group", cascade="all, delete-orphan")
    requests = relationship("GroupRequest", back_populates="group", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Group(id={self.id}, name={self.name}, subject_id={self.subject_id})>"


class GroupMember(Base):
    __tablename__ = "group_members"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id = Column(UUID(as_uuid=True), ForeignKey("groups.id"), nullable=False)
    user_name = Column(String, nullable=False)  # Store username as string (no FK constraint in microservices)
    joined_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    group = relationship("Group", back_populates="members")

    def __repr__(self):
        return f"<GroupMember(id={self.id}, group_id={self.group_id}, user_name={self.user_name})>"


class GroupRequest(Base):
    __tablename__ = "group_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id = Column(UUID(as_uuid=True), ForeignKey("groups.id"), nullable=False)
    user_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, nullable=True)

    # Relationships
    group = relationship("Group", back_populates="requests")

    def __repr__(self):
        return f"<GroupRequest(id={self.id}, group_id={self.group_id}, user_name={self.user_name}, status={self.status})>"


class CourseModule(Base):
    __tablename__ = "course_modules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), ForeignKey("subjects.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    order_index = Column(Integer, nullable=False, default=0)  # For ordering modules
    is_collapsed = Column(Boolean, default=False, nullable=False)  # Default collapsed state
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    subject = relationship("Subject", back_populates="modules")
    lessons = relationship("CourseLesson", back_populates="module", cascade="all, delete-orphan", order_by="CourseLesson.order_index")

    def __repr__(self):
        return f"<CourseModule(id={self.id}, title={self.title}, subject_id={self.subject_id})>"


class CourseLesson(Base):
    __tablename__ = "course_lessons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    module_id = Column(UUID(as_uuid=True), ForeignKey("course_modules.id"), nullable=False)
    title = Column(String, nullable=False)
    lesson_type = Column(String, nullable=False)  # lecture, quiz, video, material, exercise
    order_index = Column(Integer, nullable=False, default=0)  # For ordering lessons
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    module = relationship("CourseModule", back_populates="lessons")
    content = relationship("CourseContent", back_populates="lesson", cascade="all, delete-orphan", uselist=False)

    def __repr__(self):
        return f"<CourseLesson(id={self.id}, title={self.title}, module_id={self.module_id}, type={self.lesson_type})>"


class CourseContent(Base):
    __tablename__ = "course_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lesson_id = Column(UUID(as_uuid=True), ForeignKey("course_lessons.id"), nullable=False, unique=True)
    
    # Content fields
    text_content = Column(Text, nullable=True)  # Text/lecture content
    video_url = Column(String, nullable=True)  # YouTube/Rutube URL
    video_platform = Column(String, nullable=True)  # youtube, rutube
    material_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to material service
    test_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to test service
    
    # Additional metadata
    extra_data = Column(JSONB, nullable=True)  # Flexible JSON for additional data (renamed from metadata - reserved name)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    lesson = relationship("CourseLesson", back_populates="content")

    def __repr__(self):
        return f"<CourseContent(id={self.id}, lesson_id={self.lesson_id})>"

