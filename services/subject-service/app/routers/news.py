"""
News router - CRUD operations for news
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from datetime import timezone, timedelta

from app.database import get_db
from app.models import News, Subject
from app.schemas import NewsCreate, NewsUpdate, NewsResponse

router = APIRouter()


@router.get("", response_model=List[NewsResponse])
async def get_news(
    subject_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all news, optionally filtered by subject_id"""
    query = db.query(News)
    if subject_id:
        query = query.filter(News.subject_id == subject_id)
    news = query.order_by(News.created_at.desc()).all()
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    for item in news:
        if item.created_at and item.created_at.tzinfo is None:
            item.created_at = item.created_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
        if item.updated_at and item.updated_at.tzinfo is None:
            item.updated_at = item.updated_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    return news


@router.get("/{news_id}", response_model=NewsResponse)
async def get_news_item(news_id: UUID, db: Session = Depends(get_db)):
    """Get a specific news item"""
    news = db.query(News).filter(News.id == news_id).first()
    if not news:
        raise HTTPException(status_code=404, detail="News not found")
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    if news.created_at and news.created_at.tzinfo is None:
        news.created_at = news.created_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    if news.updated_at and news.updated_at.tzinfo is None:
        news.updated_at = news.updated_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    return news


@router.post("", response_model=NewsResponse, status_code=201)
async def create_news(news: NewsCreate, db: Session = Depends(get_db)):
    """Create a new news item"""
    # Verify subject exists
    subject = db.query(Subject).filter(Subject.id == news.subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    db_news = News(
        subject_id=news.subject_id,
        title=news.title,
        content=news.content,
        image_url=news.image_url
    )
    db.add(db_news)
    db.commit()
    db.refresh(db_news)
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    if db_news.created_at and db_news.created_at.tzinfo is None:
        db_news.created_at = db_news.created_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    if db_news.updated_at and db_news.updated_at.tzinfo is None:
        db_news.updated_at = db_news.updated_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    return db_news


@router.put("/{news_id}", response_model=NewsResponse)
async def update_news(
    news_id: UUID,
    news_update: NewsUpdate,
    db: Session = Depends(get_db)
):
    """Update a news item"""
    news = db.query(News).filter(News.id == news_id).first()
    if not news:
        raise HTTPException(status_code=404, detail="News not found")
    
    # Verify subject if provided
    if news_update.subject_id:
        subject = db.query(Subject).filter(Subject.id == news_update.subject_id).first()
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")
        news.subject_id = news_update.subject_id
    
    if news_update.title is not None:
        news.title = news_update.title
    if news_update.content is not None:
        news.content = news_update.content
    if news_update.image_url is not None:
        news.image_url = news_update.image_url
    
    db.commit()
    db.refresh(news)
    
    # Добавляем московский timezone к датам перед возвратом
    moscow_tz = timezone(timedelta(hours=3))
    if news.created_at and news.created_at.tzinfo is None:
        news.created_at = news.created_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    if news.updated_at and news.updated_at.tzinfo is None:
        news.updated_at = news.updated_at.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
    
    return news


@router.delete("/{news_id}", status_code=200)
async def delete_news(news_id: UUID, db: Session = Depends(get_db)):
    """Delete a news item"""
    news = db.query(News).filter(News.id == news_id).first()
    if not news:
        raise HTTPException(status_code=404, detail="News not found")
    
    db.delete(news)
    db.commit()
    return {"message": "News deleted successfully"}
