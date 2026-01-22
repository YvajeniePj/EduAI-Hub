"""
Users router - Get list of users for group management
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import or_

from app.database import get_db
from app.models import User

router = APIRouter()


@router.get("")
async def get_users(search: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """Get all users, optionally filtered by search query"""
    query = db.query(User)
    
    if search:
        # Search by name (case-insensitive)
        search_pattern = f"%{search}%"
        query = query.filter(
            or_(
                User.name.ilike(search_pattern)
            )
        )
    
    users = query.all()
    
    # Return only name and id
    return [{"id": str(user.id), "name": user.name} for user in users]

