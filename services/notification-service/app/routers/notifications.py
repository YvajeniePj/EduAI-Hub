"""
Notifications router - CRUD operations for notifications
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from app.database import get_db
from app.models import Notification
from app.schemas import NotificationCreate, NotificationResponse, NotificationUpdate

router = APIRouter()


@router.get("", response_model=List[NotificationResponse])
async def get_notifications(
    user_name: Optional[str] = Query(None),
    is_read: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get notifications, optionally filtered by user and read status"""
    query = db.query(Notification)
    
    if user_name:
        query = query.filter(or_(Notification.user_name == user_name, Notification.user_name.is_(None)))
    
    if is_read is not None:
        query = query.filter(Notification.is_read == is_read)
    
    notifications = query.order_by(desc(Notification.created_at)).limit(limit).all()
    return notifications


@router.get("/count", response_model=dict)
async def get_notification_count(
    user_name: str = Query(..., description="Username to get unread count for"),
    db: Session = Depends(get_db)
):
    """Get count of unread notifications for a user"""
    count = db.query(Notification).filter(
        or_(Notification.user_name == user_name, Notification.user_name.is_(None)),
        Notification.is_read == False
    ).count()
    return {"user_name": user_name, "unread_count": count}


@router.get("/{notification_id}", response_model=NotificationResponse)
async def get_notification(notification_id: UUID, db: Session = Depends(get_db)):
    """Get a specific notification"""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    return notification


import httpx
import os

SUBMISSION_SERVICE_URL = os.getenv("SUBMISSION_SERVICE_URL", "http://submission-service:8003")

@router.post("", response_model=NotificationResponse, status_code=201)
async def create_notification(notification: NotificationCreate, db: Session = Depends(get_db)):
    """Create a new notification"""
    if notification.user_name is None:
        # Fan-out: Send to all users
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{SUBMISSION_SERVICE_URL}/users")
                if response.status_code == 200:
                    users = response.json()
                    created_notifications = []
                    for user in users:
                        # Skip if user has no name (should not happen based on schema but good for safety)
                        if not user.get("name"):
                            continue
                            
                        # Exclude specific user if requested (e.g. the creator)
                        if notification.exclude_user_name and user["name"] == notification.exclude_user_name:
                            continue

                        db_notification = Notification(
                            user_name=user["name"],
                            title=notification.title,
                            message=notification.message,
                            type=notification.type,
                            related_type=notification.related_type,
                            related_id=notification.related_id
                        )
                        db.add(db_notification)
                        created_notifications.append(db_notification)
                    
                    db.commit()
                    # Return the first one or a placeholder to satisfy response_model
                    if created_notifications:
                        return created_notifications[0]
                    else:
                         # Fallback if no users found, create waiting for admin or just return
                         # But we need to return a NotificationResponse. 
                         # Let's create a "system" notification or similar if list is empty?
                         # Or just proceed to create one with user_name=None as before?
                         # Better to fall back to broadcast if fetch fails or no users?
                         pass
        except Exception as e:
            print(f"Error fetching users for broadcast: {e}")
            # Fallback to old behavior (broadcast to None) if service is down
            pass

    # Standard creation (single user OR fallback for broadcast)
    db_notification = Notification(
        user_name=notification.user_name,
        title=notification.title,
        message=notification.message,
        type=notification.type,
        related_type=notification.related_type,
        related_id=notification.related_id
    )
    db.add(db_notification)
    db.commit()
    db.refresh(db_notification)
    return db_notification


@router.put("/{notification_id}", response_model=NotificationResponse)
async def update_notification(
    notification_id: UUID,
    notification_update: NotificationUpdate,
    db: Session = Depends(get_db)
):
    """Update a notification (typically to mark as read)"""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    if notification_update.is_read is not None:
        notification.is_read = notification_update.is_read
        if notification_update.is_read and not notification.read_at:
            notification.read_at = datetime.utcnow()
        elif not notification_update.is_read:
            notification.read_at = None
    
    db.commit()
    db.refresh(notification)
    return notification


@router.post("/{notification_id}/mark-read", response_model=NotificationResponse)
async def mark_notification_read(notification_id: UUID, db: Session = Depends(get_db)):
    """Mark a notification as read"""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    notification.is_read = True
    if not notification.read_at:
        notification.read_at = datetime.utcnow()
    
    db.commit()
    db.refresh(notification)
    return notification


@router.post("/mark-all-read")
async def mark_all_read(user_name: str, db: Session = Depends(get_db)):
    """Mark all notifications as read for a user"""
    updated = db.query(Notification).filter(
        Notification.user_name == user_name,
        Notification.is_read == False
    ).update({
        "is_read": True,
        "read_at": datetime.utcnow()
    })
    db.commit()
    return {"message": f"Marked {updated} notifications as read"}


@router.delete("/{notification_id}", status_code=200)
async def delete_notification(notification_id: UUID, db: Session = Depends(get_db)):
    """Delete a notification"""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    db.delete(notification)
    db.commit()
    return {"message": "Notification deleted successfully"}

