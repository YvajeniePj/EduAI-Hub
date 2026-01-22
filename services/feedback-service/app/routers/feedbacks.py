"""
Feedbacks router - CRUD operations for feedbacks
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from uuid import UUID

from app.database import get_db
from app.models import Feedback
from app.schemas import FeedbackCreate, FeedbackResponse, FeedbackStats

router = APIRouter()


@router.get("", response_model=List[FeedbackResponse])
async def get_feedbacks(
    user_name: Optional[str] = Query(None),
    subject_id: Optional[UUID] = Query(None),
    group_id: Optional[UUID] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get feedbacks with optional filters"""
    query = db.query(Feedback)
    
    if user_name:
        query = query.filter(Feedback.user_name == user_name)
    
    if subject_id:
        query = query.filter(Feedback.subject_id == subject_id)
    
    if group_id:
        query = query.filter(Feedback.group_id == group_id)
    
    feedbacks = query.order_by(desc(Feedback.created_at)).limit(limit).all()
    return feedbacks


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(
    subject_id: Optional[UUID] = Query(None),
    group_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get aggregated feedback statistics"""
    query = db.query(Feedback)
    
    if subject_id:
        query = query.filter(Feedback.subject_id == subject_id)
    
    if group_id:
        query = query.filter(Feedback.group_id == group_id)
    
    # Calculate averages
    stats = query.with_entities(
        func.count(Feedback.id).label('total'),
        func.avg(Feedback.quality_rating).label('avg_quality'),
        func.avg(Feedback.content_rating).label('avg_content'),
        func.avg(Feedback.materials_rating).label('avg_materials'),
        func.avg(Feedback.support_rating).label('avg_support')
    ).first()
    
    if not stats or stats.total == 0:
        return FeedbackStats(
            subject_id=subject_id,
            group_id=group_id,
            total_responses=0,
            avg_quality_rating=0.0,
            avg_content_rating=0.0,
            avg_materials_rating=0.0,
            avg_support_rating=0.0,
            overall_avg=0.0
        )
    
    avg_quality = float(stats.avg_quality) if stats.avg_quality else 0.0
    avg_content = float(stats.avg_content) if stats.avg_content else 0.0
    avg_materials = float(stats.avg_materials) if stats.avg_materials else 0.0
    avg_support = float(stats.avg_support) if stats.avg_support else 0.0
    overall_avg = (avg_quality + avg_content + avg_materials + avg_support) / 4.0
    
    return FeedbackStats(
        subject_id=subject_id,
        group_id=group_id,
        total_responses=int(stats.total),
        avg_quality_rating=round(avg_quality, 2),
        avg_content_rating=round(avg_content, 2),
        avg_materials_rating=round(avg_materials, 2),
        avg_support_rating=round(avg_support, 2),
        overall_avg=round(overall_avg, 2)
    )


@router.get("/{feedback_id}", response_model=FeedbackResponse)
async def get_feedback(feedback_id: UUID, db: Session = Depends(get_db)):
    """Get a specific feedback"""
    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return feedback


@router.post("", response_model=FeedbackResponse, status_code=201)
async def create_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    """Create a new feedback"""
    db_feedback = Feedback(
        user_name=feedback.user_name,
        subject_id=feedback.subject_id,
        group_id=feedback.group_id,
        quality_rating=feedback.quality_rating,
        content_rating=feedback.content_rating,
        materials_rating=feedback.materials_rating,
        support_rating=feedback.support_rating,
        comment=feedback.comment,
        suggestions=feedback.suggestions
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback


@router.delete("/{feedback_id}", status_code=200)
async def delete_feedback(feedback_id: UUID, db: Session = Depends(get_db)):
    """Delete a feedback"""
    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    db.delete(feedback)
    db.commit()
    return {"message": "Feedback deleted successfully"}

