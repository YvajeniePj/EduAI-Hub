from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import os
import time
import logging
from datetime import datetime
from livekit import api

from app.database import get_db
from app.models import StreamingRoom
from app.schemas import RoomCreate, RoomResponse, JoinRequest, TokenResponse

router = APIRouter()
logger = logging.getLogger(__name__)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "http://livekit:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
LIVEKIT_EXTERNAL_URL = os.getenv("LIVEKIT_EXTERNAL_URL", "http://localhost:7880")

@router.post("/rooms/create", response_model=RoomResponse)
async def create_room(room_data: RoomCreate, db: Session = Depends(get_db)):
    """Create a new streaming room or return existing active one for the subject"""
    # Check if a room already exists for this subject
    existing_room = db.query(StreamingRoom).filter(
        StreamingRoom.subject_id == room_data.subject_id,
        StreamingRoom.is_active == True
    ).first()
    
    if existing_room:
        return existing_room
    
    # Generate room name if not provided
    room_name = room_data.room_name or f"room_{str(room_data.subject_id)[:8]}_{int(time.time())}"
    
    new_room = StreamingRoom(
        subject_id=room_data.subject_id,
        room_name=room_name,
        teacher_name=room_data.teacher_name,
        is_active=True
    )
    
    db.add(new_room)
    db.commit()
    db.refresh(new_room)
    
    return new_room

@router.post("/tokens/generate", response_model=TokenResponse)
async def generate_token(request: JoinRequest):
    """Generate a LiveKit access token for joining a room"""
    try:
        # Create AccessToken
        # Note: livekit-api 0.1.0 uses a different constructor or method
        # I will use the standard pattern for LiveKit tokens
        from livekit.api import AccessToken, VideoGrants
        
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(request.identity)
        token.with_name(request.identity)
        
        grants = VideoGrants(
            room_join=True,
            room=request.room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        )
        token.with_grants(grants)
        
        return TokenResponse(
            token=token.to_jwt(),
            room_name=request.room_name,
            server_url=LIVEKIT_EXTERNAL_URL
        )
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate token: {str(e)}")

@router.get("/rooms/active", response_model=List[RoomResponse])
async def get_active_rooms(db: Session = Depends(get_db)):
    """List all active streaming rooms"""
    return db.query(StreamingRoom).filter(StreamingRoom.is_active == True).all()

@router.post("/rooms/{room_name}/end")
async def end_room(room_name: str, db: Session = Depends(get_db)):
    """End a streaming session"""
    room = db.query(StreamingRoom).filter(StreamingRoom.room_name == room_name).first()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    room.is_active = False
    room.ended_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Room ended successfully"}
