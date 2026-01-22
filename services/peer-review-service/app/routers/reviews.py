"""
Review router - CRUD operations for reviews
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import os
import httpx

from app.database import get_db
from app.models import Review
from app.schemas import ReviewCreate, ReviewResponse

router = APIRouter()

# Service URLs
SUBMISSION_SERVICE_URL = os.getenv("SUBMISSION_SERVICE_URL", "http://submission-service:8003")
GAMIFICATION_SERVICE_URL = os.getenv("GAMIFICATION_SERVICE_URL", "http://gamification-service:8007")


@router.get("", response_model=List[ReviewResponse])
async def get_reviews(
    submission_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get reviews, optionally filtered by submission_id"""
    query = db.query(Review)
    if submission_id:
        query = query.filter(Review.submission_id == submission_id)
    reviews = query.all()
    return reviews


@router.get("/submissions-for-review")
async def get_submissions_for_review(
    test_id: str,
    reviewer: str,
    db: Session = Depends(get_db)
):
    """Get submissions available for review (excluding reviewer's own submissions)"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get all submissions for this test
            response = await client.get(
                f"{SUBMISSION_SERVICE_URL}/submissions",
                params={"test_id": test_id}
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch submissions")
            
            submissions = response.json()
            
            # Filter out reviewer's own submissions
            available_submissions = [
                s for s in submissions
                if s.get("user") != reviewer
            ]
            
            return available_submissions
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Submission service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching submissions: {str(e)}")


@router.get("/my-reviews")
async def get_my_reviews(
    user: str,
    test_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get reviews for user's submissions"""
    # Get user's submissions
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {"user": user}
            if test_id:
                params["test_id"] = test_id
            
            response = await client.get(
                f"{SUBMISSION_SERVICE_URL}/submissions",
                params=params
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch submissions")
            
            submissions = response.json()
            submission_ids = []
            for s in submissions:
                sub_id = s.get("id")
                if sub_id:
                    try:
                        submission_ids.append(UUID(sub_id))
                    except:
                        pass
            
            # Get reviews for these submissions
            if submission_ids:
                reviews = db.query(Review).filter(
                    Review.submission_id.in_(submission_ids)
                ).all()
            else:
                reviews = []
            
            return reviews
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Submission service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching reviews: {str(e)}")


@router.post("", response_model=ReviewResponse, status_code=201)
async def create_review(review: ReviewCreate, db: Session = Depends(get_db)):
    """Create a new review and award points to reviewer"""
    # Calculate average score
    avg_score = (review.relevance + review.structure + review.argument + review.clarity) / 4.0
    
    db_review = Review(
        submission_id=review.submission_id,
        assignment_id=review.assignment_id,
        reviewer=review.reviewer,
        relevance=review.relevance,
        structure=review.structure,
        argument=review.argument,
        clarity=review.clarity,
        avg_score=avg_score,
        comment=review.comment
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    
    # Award points to reviewer through Gamification Service
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{GAMIFICATION_SERVICE_URL}/points",
                json={
                    "user": review.reviewer,
                    "points": 1,
                    "reason": "peer_review"
                }
            )
    except Exception as e:
        # Log error but don't fail the review creation
        import logging
        logging.getLogger(__name__).warning(f"Failed to award points: {e}")
    
    return db_review

