"""
Subject router - CRUD operations for subjects
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from urllib.parse import unquote
import os
import shutil
import httpx
import logging

from app.database import get_db
from app.models import Subject
from app.schemas import SubjectCreate, SubjectResponse

router = APIRouter()


STORAGE_PATH = os.getenv("STORAGE_PATH", "/app/storage")
NOTIFICATION_SERVICE_URL = os.getenv("NOTIFICATION_SERVICE_URL", "http://notification-service:8010")
logger = logging.getLogger(__name__)


async def create_notification(user_name: str, title: str, message: str, type: str = "info", related_type: str = None, related_id: str = None, exclude_user_name: str = None):
    """Send a notification to a user or system-wide"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{NOTIFICATION_SERVICE_URL}/notifications",
                json={
                    "user_name": user_name,
                    "title": title,
                    "message": message,
                    "type": type,
                    "related_type": related_type,
                    "related_id": related_id,
                    "exclude_user_name": exclude_user_name
                }
            )
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")



@router.get("", response_model=List[SubjectResponse])
async def get_subjects(db: Session = Depends(get_db)):
    """Get all subjects"""
    subjects = db.query(Subject).all()
    return subjects


@router.post("", response_model=SubjectResponse, status_code=201)
async def create_subject(subject: SubjectCreate, db: Session = Depends(get_db), x_user_name: Optional[str] = Header(None, alias="X-User-Name")):
    """Create a new subject"""
    # Check if subject with same name already exists
    existing = db.query(Subject).filter(Subject.name == subject.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Subject with this name already exists")
    
    db_subject = Subject(name=subject.name, description=subject.description)
    db.add(db_subject)
    db.commit()
    db.refresh(db_subject)
    db.refresh(db_subject)
    
    # Notify admin or general channel about new subject
    # Sending with user_name=None makes it a broadcast notification for all users
    try:
        decoded_name = unquote(x_user_name) if x_user_name else None
    except:
        decoded_name = x_user_name

    await create_notification(
        user_name=None, 
        title="Новый курс создан", 
        message=f"Создан новый курс: {subject.name}",
        type="success",
        related_type="subject",
        related_id=str(db_subject.id),
        exclude_user_name=decoded_name
    )
    
    return db_subject


@router.put("/{subject_id}", response_model=SubjectResponse)
async def update_subject(subject_id: UUID, subject: SubjectCreate, db: Session = Depends(get_db)):
    """Update a subject"""
    db_subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not db_subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    db_subject.name = subject.name
    db_subject.description = subject.description
    db.commit()
    db.refresh(db_subject)
    return db_subject


@router.delete("/{subject_id}", status_code=200)
async def delete_subject(subject_id: UUID, db: Session = Depends(get_db)):
    """Delete a subject"""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    db.delete(subject)
    db.commit()
    return {"message": "Subject deleted successfully"}


@router.post("/{subject_id}/cover", response_model=SubjectResponse)
async def upload_cover_image(
    subject_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a cover image for a subject. Recommended size: 600x400px"""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, WebP, GIF images are allowed")
    
    # Create covers directory
    covers_dir = os.path.join(STORAGE_PATH, "covers")
    os.makedirs(covers_dir, exist_ok=True)
    
    # Save file with subject_id as name
    ext = file.filename.split(".")[-1] if file.filename else "jpg"
    filename = f"{subject_id}.{ext}"
    filepath = os.path.join(covers_dir, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store relative path in database
    subject.cover_image = f"/covers/{filename}"
    db.commit()
    db.refresh(subject)
    
    return subject


@router.get("/{subject_id}/cover")
async def get_cover_image(subject_id: UUID, db: Session = Depends(get_db)):
    """Get the cover image for a subject"""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject or not subject.cover_image:
        raise HTTPException(status_code=404, detail="Cover image not found")
    
    filepath = os.path.join(STORAGE_PATH, subject.cover_image.lstrip("/"))
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Cover image file not found")
    
    return FileResponse(filepath)
