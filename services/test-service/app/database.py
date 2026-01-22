"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://eduai:eduai_password@localhost:5432/eduai"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    from app.models import Test, Question, Keyword
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # Log error but don't fail - tables might already exist or dependencies not ready
        import logging
        logging.warning(f"Database initialization warning: {e}")

