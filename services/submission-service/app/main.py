"""
Submission Service - Handles test submissions and grading
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import submissions, users
from app.routers.submissions import user_router
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(
    title="Submission Service",
    description="Service for managing test submissions and grading",
    version="1.0.0"
)

# Create static directory if it doesn't exist
os.makedirs("static/avatars", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(submissions.router, prefix="/submissions", tags=["submissions"])
app.include_router(user_router, tags=["users"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "submission-service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

