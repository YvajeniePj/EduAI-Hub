"""
Course Structure router - Manage course modules, lessons, and content
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.database import get_db
from app.models import Subject, CourseModule, CourseLesson, CourseContent
from app.schemas import (
    CourseModuleCreate, CourseModuleUpdate, CourseModuleResponse,
    CourseLessonCreate, CourseLessonUpdate, CourseLessonResponse,
    CourseContentCreate, CourseContentUpdate, CourseContentResponse,
    CourseStructureResponse, SubjectResponse
)

router = APIRouter()


# Course Structure endpoints
@router.get("/subjects/{subject_id}/structure", response_model=CourseStructureResponse)
async def get_course_structure(subject_id: UUID, db: Session = Depends(get_db)):
    """Get full course structure with modules, lessons, and content"""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    modules = db.query(CourseModule).filter(
        CourseModule.subject_id == subject_id
    ).order_by(CourseModule.order_index).all()
    
    # Load lessons and content for each module
    for module in modules:
        lessons = db.query(CourseLesson).filter(
            CourseLesson.module_id == module.id
        ).order_by(CourseLesson.order_index).all()
        
        for lesson in lessons:
            content = db.query(CourseContent).filter(
                CourseContent.lesson_id == lesson.id
            ).first()
            lesson.content = content
        
        module.lessons = lessons
    
    return CourseStructureResponse(
        subject=subject,
        modules=modules
    )


# Module endpoints
@router.post("/subjects/{subject_id}/modules", response_model=CourseModuleResponse, status_code=201)
async def create_module(subject_id: UUID, module: CourseModuleCreate, db: Session = Depends(get_db)):
    """Create a new course module"""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Get max order_index
    max_order = db.query(CourseModule).filter(
        CourseModule.subject_id == subject_id
    ).order_by(CourseModule.order_index.desc()).first()
    
    order_index = (max_order.order_index + 1) if max_order else 0
    
    db_module = CourseModule(
        subject_id=subject_id,
        title=module.title,
        description=module.description,
        order_index=order_index,
        is_collapsed=module.is_collapsed
    )
    db.add(db_module)
    db.commit()
    db.refresh(db_module)
    return db_module


@router.put("/modules/{module_id}", response_model=CourseModuleResponse)
async def update_module(module_id: UUID, module_update: CourseModuleUpdate, db: Session = Depends(get_db)):
    """Update a course module"""
    module = db.query(CourseModule).filter(CourseModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    if module_update.title is not None:
        module.title = module_update.title
    if module_update.description is not None:
        module.description = module_update.description
    if module_update.order_index is not None:
        module.order_index = module_update.order_index
    if module_update.is_collapsed is not None:
        module.is_collapsed = module_update.is_collapsed
    
    db.commit()
    db.refresh(module)
    return module


@router.delete("/modules/{module_id}", status_code=200)
async def delete_module(module_id: UUID, db: Session = Depends(get_db)):
    """Delete a course module (cascades to lessons and content)"""
    module = db.query(CourseModule).filter(CourseModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    db.delete(module)
    db.commit()
    return {"message": "Module deleted successfully"}


# Lesson endpoints
@router.post("/modules/{module_id}/lessons", response_model=CourseLessonResponse, status_code=201)
async def create_lesson(module_id: UUID, lesson: CourseLessonCreate, db: Session = Depends(get_db)):
    """Create a new course lesson"""
    module = db.query(CourseModule).filter(CourseModule.id == module_id).first()
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    
    # Get max order_index
    max_order = db.query(CourseLesson).filter(
        CourseLesson.module_id == module_id
    ).order_by(CourseLesson.order_index.desc()).first()
    
    order_index = (max_order.order_index + 1) if max_order else 0
    
    db_lesson = CourseLesson(
        module_id=module_id,
        title=lesson.title,
        lesson_type=lesson.lesson_type,
        order_index=order_index
    )
    db.add(db_lesson)
    db.commit()
    db.refresh(db_lesson)
    return db_lesson


@router.put("/lessons/{lesson_id}", response_model=CourseLessonResponse)
async def update_lesson(lesson_id: UUID, lesson_update: CourseLessonUpdate, db: Session = Depends(get_db)):
    """Update a course lesson"""
    lesson = db.query(CourseLesson).filter(CourseLesson.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    if lesson_update.title is not None:
        lesson.title = lesson_update.title
    if lesson_update.lesson_type is not None:
        lesson.lesson_type = lesson_update.lesson_type
    if lesson_update.order_index is not None:
        lesson.order_index = lesson_update.order_index
    
    db.commit()
    db.refresh(lesson)
    return lesson


@router.delete("/lessons/{lesson_id}", status_code=200)
async def delete_lesson(lesson_id: UUID, db: Session = Depends(get_db)):
    """Delete a course lesson (cascades to content)"""
    lesson = db.query(CourseLesson).filter(CourseLesson.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    db.delete(lesson)
    db.commit()
    return {"message": "Lesson deleted successfully"}


# Content endpoints
@router.post("/lessons/{lesson_id}/content", response_model=CourseContentResponse, status_code=201)
async def create_content(lesson_id: UUID, content: CourseContentCreate, db: Session = Depends(get_db)):
    """Create or update content for a lesson"""
    lesson = db.query(CourseLesson).filter(CourseLesson.id == lesson_id).first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Check if content already exists
    existing = db.query(CourseContent).filter(CourseContent.lesson_id == lesson_id).first()
    if existing:
        # Update existing
        if content.text_content is not None:
            existing.text_content = content.text_content
        if content.video_url is not None:
            existing.video_url = content.video_url
        if content.video_platform is not None:
            existing.video_platform = content.video_platform
        if content.material_id is not None:
            existing.material_id = content.material_id
        if content.test_id is not None:
            existing.test_id = content.test_id
        if content.extra_data is not None:
            existing.extra_data = content.extra_data
        
        db.commit()
        db.refresh(existing)
        return existing
    
    # Create new
    db_content = CourseContent(
        lesson_id=lesson_id,
        text_content=content.text_content,
        video_url=content.video_url,
        video_platform=content.video_platform,
        material_id=content.material_id,
        test_id=content.test_id,
        extra_data=content.extra_data
    )
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content


@router.put("/content/{content_id}", response_model=CourseContentResponse)
async def update_content(content_id: UUID, content_update: CourseContentUpdate, db: Session = Depends(get_db)):
    """Update course content"""
    content = db.query(CourseContent).filter(CourseContent.id == content_id).first()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    if content_update.text_content is not None:
        content.text_content = content_update.text_content
    if content_update.video_url is not None:
        content.video_url = content_update.video_url
    if content_update.video_platform is not None:
        content.video_platform = content_update.video_platform
    if content_update.material_id is not None:
        content.material_id = content_update.material_id
    if content_update.test_id is not None:
        content.test_id = content_update.test_id
    if content_update.extra_data is not None:
        content.extra_data = content_update.extra_data
    
    db.commit()
    db.refresh(content)
    return content


@router.get("/lessons/{lesson_id}/content", response_model=CourseContentResponse)
async def get_content(lesson_id: UUID, db: Session = Depends(get_db)):
    """Get content for a lesson"""
    content = db.query(CourseContent).filter(CourseContent.lesson_id == lesson_id).first()
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    return content
