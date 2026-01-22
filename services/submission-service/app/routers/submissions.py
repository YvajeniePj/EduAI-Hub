"""
Submission router - Handles test submissions
"""
from fastapi import APIRouter, HTTPException, Depends, Query, File, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from datetime import datetime
import httpx
import os
import shutil
from pathlib import Path

from app.database import get_db
from app.models import Submission, Answer, User
from app.schemas import (
    SubmissionCreate,
    SubmissionUpdate,
    SubmissionResponse,
    SubmissionResults,
    UserCreate,
    UserUpdate,
    UserResponse,
)
from app.services.grading import grade_multiple_choice, grade_keyword_based

router = APIRouter()
user_router = APIRouter()

TEST_SERVICE_URL = os.getenv("TEST_SERVICE_URL", "http://test-service:8002")
GAMIFICATION_SERVICE_URL = os.getenv("GAMIFICATION_SERVICE_URL", "http://gamification-service:8007")


async def get_test_from_service(test_id: UUID) -> dict:
    """Fetch test data from test service"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{TEST_SERVICE_URL}/tests/{test_id}")
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Test not found")
        return response.json()


async def award_points(user: str, points: int):
    """Award points to user via gamification service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{GAMIFICATION_SERVICE_URL}/points/award",
                json={"user": user, "points": points}
            )
    except Exception as e:
        # Log but don't fail if gamification service is unavailable
        print(f"Failed to award points: {e}")


@router.get("", response_model=List[SubmissionResponse])
async def get_submissions(
    test_id: Optional[UUID] = Query(None),
    user: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get submissions, optionally filtered by test_id or user"""
    query = db.query(Submission)
    if test_id:
        query = query.filter(Submission.test_id == test_id)
    if user:
        query = query.filter(Submission.user == user)
    submissions = query.all()
    return submissions


# Users (no /submissions prefix)
@user_router.get("/users", response_model=List[UserResponse])
async def get_users(search: Optional[str] = Query(None), db: Session = Depends(get_db)):
    query = db.query(User)
    if search:
        # Search by name (case-insensitive)
        search_pattern = f"%{search}%"
        query = query.filter(User.name.ilike(search_pattern))
    return query.all()


@user_router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: UUID, db: Session = Depends(get_db)):
    user_obj = db.query(User).filter(User.id == user_id).first()
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    return user_obj


@user_router.get("/users/by-name/{name}", response_model=UserResponse)
async def get_user_by_name(name: str, db: Session = Depends(get_db)):
    user_obj = db.query(User).filter(User.name == name).first()
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    return user_obj


@user_router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.name == user.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    db_user = User(name=user.name, role=user.role, avatar_url=user.avatar_url)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@user_router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: UUID, user_update: UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_update.name is not None:
        # Check if new name is taken
        if user_update.name != db_user.name:
            existing = db.query(User).filter(User.name == user_update.name).first()
            if existing:
                raise HTTPException(status_code=400, detail="Username already taken")
        db_user.name = user_update.name
    
    if user_update.avatar_url is not None:
        db_user.avatar_url = user_update.avatar_url

    if user_update.role is not None:
        db_user.role = user_update.role
        
    db.commit()
    db.refresh(db_user)
    return db_user


@user_router.post("/users/{user_id}/avatar")
async def upload_avatar(user_id: UUID, file: UploadFile = File(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create avatars directory if it doesn't exist
    avatar_dir = Path("static/avatars")
    avatar_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_extension = Path(file.filename).suffix
    file_name = f"{user_id}{file_extension}"
    file_path = avatar_dir / file_name
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Update user in DB
    avatar_url = f"/static/avatars/{file_name}"
    db_user.avatar_url = avatar_url
    db.commit()
    
    return {"avatar_url": avatar_url}


@router.get("/{submission_id}", response_model=SubmissionResponse)
async def get_submission(submission_id: UUID, db: Session = Depends(get_db)):
    """Get a submission by ID"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    return submission


@router.post("", response_model=SubmissionResponse, status_code=201)
async def create_submission(
    submission: SubmissionCreate,
    db: Session = Depends(get_db)
):
    """Create a new submission (start a test)"""
    # Verify test exists
    test_data = await get_test_from_service(submission.test_id)
    
    # Create submission
    db_submission = Submission(
        test_id=submission.test_id,
        user=submission.user,
        assignment=submission.assignment or test_data.get("assignment_id"),
        total_max=sum(q.get("max_points", 0) for q in test_data.get("questions", []))
    )
    db.add(db_submission)
    db.flush()
    
    # Create initial answers
    for answer_data in submission.answers:
        db_answer = Answer(
            submission_id=db_submission.id,
            question_id=answer_data.question_id,
            answer=answer_data.answer
        )
        db.add(db_answer)
    
    db.commit()
    db.refresh(db_submission)
    return db_submission


@router.put("/{submission_id}", response_model=SubmissionResponse)
async def update_submission(
    submission_id: UUID,
    submission_update: SubmissionUpdate,
    db: Session = Depends(get_db)
):
    """Update submission answers"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    if submission.is_finished == "true":
        raise HTTPException(status_code=400, detail="Submission already finished")
    
    # Update or create answers
    for answer_data in submission_update.answers:
        existing_answer = db.query(Answer).filter(
            Answer.submission_id == submission_id,
            Answer.question_id == answer_data.question_id
        ).first()
        
        if existing_answer:
            existing_answer.answer = answer_data.answer
        else:
            db_answer = Answer(
                submission_id=submission_id,
                question_id=answer_data.question_id,
                answer=answer_data.answer
            )
            db.add(db_answer)
    
    db.commit()
    db.refresh(submission)
    return submission


@router.post("/{submission_id}/finish", response_model=SubmissionResponse)
async def finish_submission(
    submission_id: UUID,
    use_ai: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Finish a submission and calculate scores"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    if submission.is_finished == "true":
        raise HTTPException(status_code=400, detail="Submission already finished")
    
    # Get test data
    test_data = await get_test_from_service(submission.test_id)
    test_type = test_data.get("test_type")
    questions = test_data.get("questions", [])
    
    total_score = 0
    per_q_results = []
    
    # Grade each answer
    for q in questions:
        q_id = q.get("question_id")
        answer_obj = db.query(Answer).filter(
            Answer.submission_id == submission_id,
            Answer.question_id == q_id
        ).first()
        
        if not answer_obj:
            answer_text = ""
        else:
            answer_text = answer_obj.answer
        
        max_points = q.get("max_points", 0)
        
        if test_type == "multiple_choice":
            correct_answer = q.get("correct_answer", "")
            score, details = await grade_multiple_choice(answer_text, correct_answer, max_points)
            answer_obj.score = score
            answer_obj.final_score = score
            answer_obj.details = details
            total_score += score
            
            per_q_results.append({
                "question_id": q_id,
                "title": q.get("title"),
                "answer": answer_text,
                "score": score,
                "max_points": max_points,
                "details": details
            })
        
        elif test_type == "keyword_based":
            keywords = q.get("keywords", [])
            
            # Always use AI feedback for keyword-based tests
            kw_score, final_score, details, ai_feedback_data = await grade_keyword_based(
                answer_text, keywords, max_points, submission.test_id, q_id, q.get("title")
            )
            
            if answer_obj:
                answer_obj.score = kw_score
                answer_obj.final_score = final_score
                answer_obj.ai_score = ai_feedback_data.get("recommended_score") if ai_feedback_data else None
                answer_obj.ai_feedback = ai_feedback_data  # Store full feedback object
                answer_obj.details = details
            
            total_score += final_score
            
            per_q_results.append({
                "question_id": q_id,
                "title": q.get("title"),
                "answer": answer_text,
                "kw_score": kw_score,
                "final_score": final_score,
                "max_points": max_points,
                "ai_feedback": ai_feedback_data,  # Full feedback object
                "details": details
            })
    
    # Update submission
    submission.total_score = total_score
    submission.finished_at = datetime.utcnow()
    submission.is_finished = "true"
    
    # Calculate points awarded (1 point per ~10% of score, minimum 1)
    if submission.total_max > 0:
        percentage = (total_score / submission.total_max) * 100
        points_awarded = max(1, int(percentage / 10))
    else:
        points_awarded = 0
    
    submission.points_awarded = points_awarded
    
    # Award points via gamification service
    await award_points(submission.user, points_awarded)
    
    db.commit()
    db.refresh(submission)
    return submission


@router.get("/{submission_id}/results", response_model=SubmissionResults)
async def get_submission_results(
    submission_id: UUID,
    db: Session = Depends(get_db)
):
    """Get detailed results for a submission"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # Get test data for question details
    test_data = await get_test_from_service(submission.test_id)
    questions = test_data.get("questions", [])
    
    # Build per-question results
    per_q_results = []
    for answer in submission.answers:
        q_data = next((q for q in questions if q.get("question_id") == answer.question_id), None)
        per_q_results.append({
            "question_id": answer.question_id,
            "title": q_data.get("title") if q_data else "",
            "answer": answer.answer,
            "score": answer.final_score or answer.score,
            "max_points": q_data.get("max_points", 0) if q_data else 0,
            "ai_score": answer.ai_score,
            "ai_feedback": answer.ai_feedback,
            "details": answer.details
        })
    
    return SubmissionResults(
        submission=submission,
        per_question_results=per_q_results
    )

