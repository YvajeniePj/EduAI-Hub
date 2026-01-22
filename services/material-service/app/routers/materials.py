"""
Material router - CRUD operations for materials
"""
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import os
import shutil
import time
import httpx
import logging
import json
from urllib.parse import unquote

from app.database import get_db
from app.models import Material
from app.schemas import MaterialCreate, MaterialResponse
from app.services.text_extraction import extract_text_from_file

router = APIRouter()
logger = logging.getLogger(__name__)

STORAGE_PATH = os.getenv("STORAGE_PATH", "/app/storage")
STORAGE_PATH = os.getenv("STORAGE_PATH", "/app/storage")
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8008")
NOTIFICATION_SERVICE_URL = os.getenv("NOTIFICATION_SERVICE_URL", "http://notification-service:8010")

async def create_notification(user_name: str, title: str, message: str, type: str = "info", related_type: str = None, related_id: str = None, exclude_user_name: str = None):
    """Send a notification"""
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


@router.get("", response_model=List[MaterialResponse])
async def get_materials(
    subject_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all materials, optionally filtered by subject_id"""
    query = db.query(Material)
    if subject_id:
        query = query.filter(Material.subject_id == subject_id)
    materials = query.all()
    return materials


@router.get("/{material_id}", response_model=MaterialResponse)
async def get_material(material_id: UUID, db: Session = Depends(get_db)):
    """Get a material by ID"""
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    return material


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing unsafe characters"""
    import re
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    safe = safe.strip('. ')
    return safe if safe else "file"


@router.post("", response_model=MaterialResponse, status_code=201)
async def create_material(
    subject_id: UUID = Form(...),
    file: UploadFile = File(...),
    note: Optional[str] = Form(None),
    allowed_groups: Optional[str] = Form(None),
    uploader: str = Form("anonymous"),
    x_user_name: Optional[str] = Header(None, alias="X-User-Name"),
    db: Session = Depends(get_db)
):
    """Upload a new material"""
    if x_user_name:
        uploader = unquote(x_user_name)
    # Ensure storage directory exists
    os.makedirs(STORAGE_PATH, exist_ok=True)
    
    # Create subject-specific directory
    subject_dir = os.path.join(STORAGE_PATH, str(subject_id))
    os.makedirs(subject_dir, exist_ok=True)
    
    # Generate safe filename
    original_name = file.filename or "file"
    safe_name = safe_filename(original_name)
    
    # Avoid overwriting: add suffix if file exists
    base, ext = os.path.splitext(safe_name)
    file_path = os.path.join(subject_dir, safe_name)
    k = 1
    while os.path.exists(file_path):
        safe_name = f"{base}({k}){ext}"
        file_path = os.path.join(subject_dir, safe_name)
        k += 1
    
    # Save file
    try:
        with open(file_path, "wb") as out:
            content = await file.read()
            out.write(content)
        
        file_size = len(content)
        mime_type = file.content_type or "application/octet-stream"
        
        # Create material record
        material = Material(
            subject_id=subject_id,
            name=safe_name,
            original_name=original_name,
            path=file_path,
            size=file_size,
            mime_type=mime_type,
            uploader=uploader,
            note=note.strip() if note else None,
            allowed_groups=json.loads(allowed_groups) if allowed_groups else None
        )
        
        db.add(material)
        db.commit()
        db.refresh(material)
        
        # Notify about new material
        await create_notification(
            user_name=None, 
            title="Новый материал загружен", 
            message=f"Загружен новый материал: {original_name}",
            type="info",
            related_type="material",
            related_id=str(material.id),
            exclude_user_name=uploader
        )
        
        return material
    except Exception as e:
        # Clean up file if database operation fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.get("/{material_id}/text")
async def get_material_text(material_id: UUID, db: Session = Depends(get_db)):
    """Extract and return text from a material file"""
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    if not os.path.exists(material.path):
        raise HTTPException(status_code=404, detail="Material file not found on disk")
    
    text = extract_text_from_file(material.path, material.mime_type)
    
    if text.startswith("Error") or text.startswith("Формат файла"):
        raise HTTPException(status_code=400, detail=text)
    
    return {"text": text}


@router.post("/{material_id}/annotate")
async def create_annotation(
    material_id: UUID,
    db: Session = Depends(get_db)
):
    """Create AI annotations for a material in both Russian and English"""
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    # Extract text from file
    if not os.path.exists(material.path):
        raise HTTPException(status_code=404, detail="Material file not found on disk")
    
    text = extract_text_from_file(material.path, material.mime_type)
    
    if text.startswith("Error") or text.startswith("Формат файла"):
        raise HTTPException(status_code=400, detail=f"Cannot extract text: {text}")
    
    if not text or len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Not enough text for annotation (minimum 50 characters)")
    
    # Call AI Service for both language annotations
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout for two requests
            # Generate Russian annotation
            response_ru = await client.post(
                f"{AI_SERVICE_URL}/ai/annotate",
                json={
                    "text": text[:4000],  # Limit text length
                    "filename": material.original_name or material.name,
                    "language": "ru"
                }
            )
            
            # Generate English annotation
            response_en = await client.post(
                f"{AI_SERVICE_URL}/ai/annotate",
                json={
                    "text": text[:4000],  # Limit text length
                    "filename": material.original_name or material.name,
                    "language": "en"
                }
            )
            
            annotation_ru = ""
            annotation_en = ""
            
            if response_ru.status_code == 200:
                result_ru = response_ru.json()
                annotation_ru = result_ru.get("annotation", "")
            else:
                logger.warning(f"Failed to generate Russian annotation: {response_ru.status_code}")
            
            if response_en.status_code == 200:
                result_en = response_en.json()
                annotation_en = result_en.get("annotation", "")
            else:
                logger.warning(f"Failed to generate English annotation: {response_en.status_code}")
            
            if not annotation_ru and not annotation_en:
                raise HTTPException(status_code=500, detail="Failed to generate annotations in both languages")
            
            # Update material with both annotations
            material.annotation_ru = annotation_ru
            material.annotation_en = annotation_en
            # Keep backward compatibility
            material.annotation = annotation_ru or annotation_en
            db.commit()
            db.refresh(material)
            
            return {
                "annotation_ru": annotation_ru,
                "annotation_en": annotation_en
            }
    except httpx.RequestError as e:
        logger.error(f"Request error to AI Service: {e}")
        raise HTTPException(status_code=503, detail=f"AI Service unavailable: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating annotation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating annotation: {str(e)}")


@router.delete("/{material_id}", status_code=200)
async def delete_material(material_id: UUID, db: Session = Depends(get_db)):
    """Delete a material"""
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    # Delete file if exists
    if os.path.exists(material.path):
        try:
            os.remove(material.path)
        except Exception as e:
            # Log error but continue with database deletion
            pass
    
    db.delete(material)
    db.commit()
    return {"message": "Material deleted successfully"}


@router.get("/{material_id}/download")
async def download_material(material_id: UUID, db: Session = Depends(get_db)):
    """Download a material file"""
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    if not os.path.exists(material.path):
        raise HTTPException(status_code=404, detail="Material file not found on disk")
        
    return FileResponse(
        path=material.path, 
        filename=material.original_name or material.name,
        media_type=material.mime_type
    )


