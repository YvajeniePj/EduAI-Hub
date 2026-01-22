"""
Points router - Manages points and leaderboard
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from uuid import UUID
import csv
import io

from app.database import get_db
from app.models import Points
from app.schemas import PointsAward, PointsResponse, LeaderboardEntry

router = APIRouter()


@router.get("", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    subject_id: Optional[UUID] = Query(None),
    limit: Optional[int] = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get leaderboard, optionally filtered by subject_id"""
    # Aggregate points by user (sum across all subjects or filter by subject)
    if subject_id:
        query = db.query(
            Points.user,
            func.sum(Points.points).label('total_points')
        ).filter(Points.subject_id == subject_id).group_by(Points.user)
    else:
        query = db.query(
            Points.user,
            func.sum(Points.points).label('total_points')
        ).group_by(Points.user)
    
    results = query.order_by(func.sum(Points.points).desc()).limit(limit).all()
    
    # Convert to response format
    leaderboard = []
    for rank, (user, total_points) in enumerate(results, start=1):
        leaderboard.append({
            "rank": rank,
            "user": user,
            "points": int(total_points) if total_points else 0
        })
    
    return leaderboard


@router.get("/export")
async def export_leaderboard(
    subject_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Export leaderboard as CSV"""
    # Get leaderboard data
    if subject_id:
        query = db.query(
            Points.user,
            func.sum(Points.points).label('total_points')
        ).filter(Points.subject_id == subject_id).group_by(Points.user)
    else:
        query = db.query(
            Points.user,
            func.sum(Points.points).label('total_points')
        ).group_by(Points.user)
    
    results = query.order_by(func.sum(Points.points).desc()).all()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "User", "Points"])
    
    for rank, (user, total_points) in enumerate(results, start=1):
        writer.writerow([rank, user, int(total_points) if total_points else 0])
    
    csv_content = output.getvalue()
    output.close()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=leaderboard.csv"}
    )


@router.get("/{username}", response_model=PointsResponse)
async def get_user_points(
    username: str,
    subject_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get points for a specific user"""
    query = db.query(Points).filter(Points.user == username)
    if subject_id:
        query = query.filter(Points.subject_id == subject_id)
    points = query.first()
    if not points:
        # Create if doesn't exist
        points = Points(user=username, subject_id=subject_id, points=0)
        db.add(points)
        db.commit()
        db.refresh(points)
    return points


@router.post("/award", response_model=PointsResponse)
async def award_points(award: PointsAward, db: Session = Depends(get_db)):
    """Award points to a user"""
    # Find or create points record
    query = db.query(Points).filter(Points.user == award.user)
    if award.subject_id:
        query = query.filter(Points.subject_id == award.subject_id)
    points = query.first()
    
    if points:
        points.points += award.points
    else:
        points = Points(
            user=award.user,
            subject_id=award.subject_id,
            points=award.points
        )
        db.add(points)
    
    db.commit()
    db.refresh(points)
    return points

