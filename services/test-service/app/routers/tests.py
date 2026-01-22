"""
Test router - CRUD operations for tests
"""
from fastapi import APIRouter, HTTPException, Depends, Query, Header
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from datetime import timezone, timedelta
import httpx
import logging
import os

from app.database import get_db
from app.models import Test
from app.schemas import TestCreate, TestUpdate, TestResponse

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


@router.get("", response_model=List[TestResponse])
async def get_tests(
    subject_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all tests, optionally filtered by subject_id"""
    query = db.query(Test)
    if subject_id:
        query = query.filter(Test.subject_id == subject_id)
    tests = query.all()
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    for test in tests:
        if test.due_date and test.due_date.tzinfo is None:
            test.due_date = test.due_date.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
        if test.available_until and test.available_until.tzinfo is None:
            test.available_until = test.available_until.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    return tests


@router.get("/{test_id}", response_model=TestResponse)
async def get_test(test_id: UUID, db: Session = Depends(get_db)):
    """Get a test by ID"""
    test = db.query(Test).filter(Test.id == test_id).first()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    if test.due_date and test.due_date.tzinfo is None:
        test.due_date = test.due_date.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    if test.available_until and test.available_until.tzinfo is None:
        test.available_until = test.available_until.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    return test


@router.post("", response_model=TestResponse, status_code=201)
async def create_test(test: TestCreate, db: Session = Depends(get_db), x_user_name: Optional[str] = Header(None, alias="X-User-Name")):
    """Create a new test with questions"""
    from app.models import Question, Keyword
    
    # Create test
    db_test = Test(
        subject_id=test.subject_id,
        title=test.title,
        description=test.description,
        assignment_id=test.assignment_id,
        test_type=test.test_type,
        due_date=test.due_date,
        available_until=test.available_until,
        time_limit_minutes=test.time_limit_minutes,
        ai_generated=str(test.ai_generated) if test.ai_generated else "false",
        allowed_groups=test.allowed_groups
    )
    db.add(db_test)
    db.flush()  # Get the test ID
    
    # Create questions
    if test.questions:
        for q_data in test.questions:
            db_question = Question(
                test_id=db_test.id,
                question_id=q_data.question_id,
                title=q_data.title,
                max_points=q_data.max_points,
                test_type=q_data.test_type,
                options=q_data.options,
                correct_answer=q_data.correct_answer
            )
            db.add(db_question)
            db.flush()
            
            # Create keywords for keyword_based questions
            if q_data.test_type.value == "keyword_based" and q_data.keywords:
                for kw_data in q_data.keywords:
                    db_keyword = Keyword(
                        question_id=db_question.id,
                        word=kw_data.word,
                        points=kw_data.points
                    )
                    db.add(db_keyword)
    
    db.commit()
    db.refresh(db_test)
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    if db_test.due_date and db_test.due_date.tzinfo is None:
        db_test.due_date = db_test.due_date.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    if db_test.available_until and db_test.available_until.tzinfo is None:
        db_test.available_until = db_test.available_until.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    if db_test.available_until and db_test.available_until.tzinfo is None:
        db_test.available_until = db_test.available_until.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    # Format due date for notification
    due_date_str = ""
    if test.due_date:
        # Ensure timezone awareness (assuming input might be naive UTC or already aware)
        dd = test.due_date
        if dd.tzinfo is None:
            dd = dd.replace(tzinfo=timezone.utc)
        
        # Convert to Moscow time for user display
        moscow_tz = timezone(timedelta(hours=3))
        dd_msk = dd.astimezone(moscow_tz)
        due_date_str = f". Дедлайн: {dd_msk.strftime('%d.%m.%Y %H:%M')}"

    # Notify admin/all about new test
    try:
        decoded_name = unquote(x_user_name) if x_user_name else None
    except:
        decoded_name = x_user_name

    await create_notification(
        user_name=None, 
        title="Новый тест доступен", 
        message=f"Добавлен новый тест: {test.title}{due_date_str}",
        type="info",
        related_type="test",
        related_id=str(db_test.id),
        exclude_user_name=decoded_name
    )

    return db_test


@router.put("/{test_id}", response_model=TestResponse)
async def update_test(
    test_id: UUID,
    test_update: TestUpdate,
    db: Session = Depends(get_db)
):
    """Update a test"""
    test = db.query(Test).filter(Test.id == test_id).first()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    # Update fields
    if test_update.title is not None:
        test.title = test_update.title
    if test_update.description is not None:
        test.description = test_update.description
    if test_update.due_date is not None:
        test.due_date = test_update.due_date
    if test_update.available_until is not None:
        test.available_until = test_update.available_until
    if test_update.time_limit_minutes is not None:
        test.time_limit_minutes = test_update.time_limit_minutes
    if test_update.allowed_groups is not None:
        test.allowed_groups = test_update.allowed_groups
    
    db.commit()
    db.refresh(test)
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    if test.due_date and test.due_date.tzinfo is None:
        test.due_date = test.due_date.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    if test.available_until and test.available_until.tzinfo is None:
        test.available_until = test.available_until.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    return test


@router.delete("/{test_id}", status_code=200)
async def delete_test(test_id: UUID, db: Session = Depends(get_db)):
    """Delete a test"""
    test = db.query(Test).filter(Test.id == test_id).first()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    db.delete(test)
    db.commit()
    return {"message": "Test deleted successfully"}

