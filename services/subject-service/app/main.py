"""
Subject Service - Manages subjects/courses
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os

from app.database import get_db, init_db
from app.models import Subject
from app.schemas import SubjectCreate, SubjectResponse
from app.routers import subjects, news, groups, course_structure

app = FastAPI(
    title="Subject Service",
    description="Service for managing subjects/courses, groups, and news",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(subjects.router, prefix="/subjects", tags=["subjects"])
app.include_router(news.router, prefix="/news", tags=["news"])
app.include_router(groups.router, prefix="/groups", tags=["groups"])
app.include_router(course_structure.router, tags=["course-structure"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "subject-service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

