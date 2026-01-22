"""
Video router - CRUD operations for videos
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Header
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from urllib.parse import unquote
import json
import httpx
import logging
import os

from app.database import get_db
from app.models import Video
from app.schemas import VideoCreate, VideoResponse
from app.services.video_parser import get_video_info

router = APIRouter()
logger = logging.getLogger(__name__)

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


@router.get("", response_model=List[VideoResponse])
async def get_videos(
    subject_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all videos, optionally filtered by subject_id"""
    query = db.query(Video)
    if subject_id:
        query = query.filter(Video.subject_id == subject_id)
    videos = query.all()
    return videos


@router.post("", response_model=VideoResponse, status_code=201)
async def create_video(video: VideoCreate, db: Session = Depends(get_db), x_user_name: Optional[str] = Header(None, alias="X-User-Name")):
    """Create a new video with automatic parsing"""
    # Parse video URL to get information
    video_info = await get_video_info(video.url)
    
    # Use parsed title if available, otherwise use provided title
    title = video.title
    if video_info and video_info.get("title"):
        title = video_info["title"]
    
    # Store video_info as JSON (SQLAlchemy JSON type handles serialization)
    db_video = Video(
        subject_id=video.subject_id,
        url=video.url,
        title=title,
        note=video.note,
        uploader=video.uploader,
        video_info=video_info,
        allowed_groups=video.allowed_groups
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    # Notify about new video
    try:
        decoded_name = unquote(x_user_name) if x_user_name else None
    except:
        decoded_name = x_user_name

    await create_notification(
        user_name=None, 
        title="Новое видео добавлено", 
        message=f"Добавлено новое видео: {title}",
        type="info",
        related_type="video",
        related_id=str(db_video.id),
        exclude_user_name=decoded_name
    )
    
    # Return with parsed video_info
    return {
        "id": db_video.id,
        "subject_id": db_video.subject_id,
        "url": db_video.url,
        "title": db_video.title,
        "note": db_video.note,
        "uploader": db_video.uploader,
        "created_at": db_video.created_at,
        "video_info": video_info,
        "allowed_groups": db_video.allowed_groups
    }


@router.delete("/{video_id}", status_code=200)
async def delete_video(video_id: UUID, db: Session = Depends(get_db)):
    """Delete a video"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    db.delete(video)
    db.commit()
    return {"message": "Video deleted successfully"}

