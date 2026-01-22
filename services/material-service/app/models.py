"""
Database models for Material Service
"""
from sqlalchemy import Column, String, DateTime, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.database import Base


class Material(Base):
    __tablename__ = "materials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), nullable=False)  # No foreign key in microservices
    name = Column(String, nullable=False)
    original_name = Column(String, nullable=True)
    path = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    uploader = Column(String, nullable=False)
    note = Column(String, nullable=True)
    annotation_ru = Column(String, nullable=True)  # AI-generated annotation in Russian
    annotation_en = Column(String, nullable=True)  # AI-generated annotation in English
    annotation = Column(String, nullable=True)  # Deprecated: kept for backward compatibility
    allowed_groups = Column(JSON, nullable=True)  # List of Group IDs allowed to view this material
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Material(id={self.id}, name={self.name})>"

