"""
Question router - CRUD operations for questions
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID

from app.database import get_db
from app.models import Question, Keyword, Test
from app.schemas import QuestionCreate, QuestionResponse

router = APIRouter()


@router.post("/{test_id}/questions", response_model=QuestionResponse, status_code=201)
async def create_question(
    test_id: UUID,
    question: QuestionCreate,
    db: Session = Depends(get_db)
):
    """Add a question to a test"""
    # Verify test exists
    test = db.query(Test).filter(Test.id == test_id).first()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    
    # Create question
    db_question = Question(
        test_id=test_id,
        question_id=question.question_id,
        title=question.title,
        max_points=question.max_points,
        test_type=question.test_type,
        options=question.options,
        correct_answer=question.correct_answer
    )
    db.add(db_question)
    db.flush()
    
    # Create keywords if needed
    if question.test_type.value == "keyword_based" and question.keywords:
        for kw_data in question.keywords:
            db_keyword = Keyword(
                question_id=db_question.id,
                word=kw_data.word,
                points=kw_data.points
            )
            db.add(db_keyword)
    
    db.commit()
    db.refresh(db_question)
    return db_question


@router.put("/{test_id}/questions/{question_id}", response_model=QuestionResponse)
async def update_question(
    test_id: UUID,
    question_id: UUID,
    question: QuestionCreate,
    db: Session = Depends(get_db)
):
    """Update a question"""
    db_question = db.query(Question).filter(
        Question.id == question_id,
        Question.test_id == test_id
    ).first()
    if not db_question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Update question fields
    db_question.question_id = question.question_id
    db_question.title = question.title
    db_question.max_points = question.max_points
    db_question.test_type = question.test_type
    db_question.options = question.options
    db_question.correct_answer = question.correct_answer
    
    # Delete old keywords and create new ones
    db.query(Keyword).filter(Keyword.question_id == question_id).delete()
    if question.test_type.value == "keyword_based" and question.keywords:
        for kw_data in question.keywords:
            db_keyword = Keyword(
                question_id=db_question.id,
                word=kw_data.word,
                points=kw_data.points
            )
            db.add(db_keyword)
    
    db.commit()
    db.refresh(db_question)
    return db_question


@router.delete("/{test_id}/questions/{question_id}", status_code=200)
async def delete_question(
    test_id: UUID,
    question_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a question"""
    question = db.query(Question).filter(
        Question.id == question_id,
        Question.test_id == test_id
    ).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    db.delete(question)
    db.commit()
    return {"message": "Question deleted successfully"}

