"""
AI Service - Handles AI functions using GigaChat
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import ai

app = FastAPI(
    title="AI Service",
    description="Service for AI functions using GigaChat",
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
app.include_router(ai.router, prefix="/ai", tags=["ai"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

