"""
Analytics router - Analytics and monitoring endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta

from app.database import get_db
from app.models import UserActivity, StudentProgress
from app.schemas import (
    UserActivityCreate, UserActivityResponse,
    StudentProgressResponse, AnalyticsReport, ActivityStats
)

router = APIRouter()


@router.post("/activities", response_model=UserActivityResponse, status_code=201)
async def create_activity(activity: UserActivityCreate, db: Session = Depends(get_db)):
    """Create a new user activity record"""
    db_activity = UserActivity(
        user_name=activity.user_name,
        action_type=activity.action_type,
        resource_type=activity.resource_type,
        resource_id=activity.resource_id,
        session_duration=activity.session_duration
    )
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)
    
    # Update student progress if needed
    # This could be done asynchronously in production
    return db_activity


@router.get("/activities", response_model=List[UserActivityResponse])
async def get_activities(
    user_name: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get user activities with optional filters"""
    query = db.query(UserActivity)
    
    if user_name:
        query = query.filter(UserActivity.user_name == user_name)
    
    if action_type:
        query = query.filter(UserActivity.action_type == action_type)
    
    activities = query.order_by(desc(UserActivity.created_at)).limit(limit).all()
    return activities


@router.get("/progress", response_model=List[StudentProgressResponse])
async def get_progress(
    user_name: Optional[str] = Query(None),
    subject_id: Optional[UUID] = Query(None),
    group_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get student progress with optional filters"""
    query = db.query(StudentProgress)
    
    if user_name:
        query = query.filter(StudentProgress.user_name == user_name)
    
    if subject_id:
        query = query.filter(StudentProgress.subject_id == subject_id)
    
    if group_id:
        query = query.filter(StudentProgress.group_id == group_id)
    
    progress_list = query.all()
    
    # If no progress records, create basic ones from activities
    if not progress_list:
        activity_query = db.query(UserActivity)
        if user_name:
            activity_query = activity_query.filter(UserActivity.user_name == user_name)
        
        activities = activity_query.all()
        
        if activities:
            # Group by user
            user_activities = {}
            for activity in activities:
                if activity.user_name not in user_activities:
                    user_activities[activity.user_name] = []
                user_activities[activity.user_name].append(activity)
            
            # Create progress records from activities
            for username, user_acts in user_activities.items():
                logins = sum(1 for a in user_acts if a.action_type == "login")
                test_actions = sum(1 for a in user_acts if a.action_type in ["test_start", "test_finish"])
                total_time = sum(a.session_duration or 0 for a in user_acts)
                
                progress = StudentProgress(
                    user_name=username,
                    subject_id=subject_id,
                    group_id=group_id,
                    tests_completed=test_actions // 2,
                    tests_total=0,
                    average_score=None,
                    total_time_seconds=total_time,
                    login_count=logins,
                    materials_viewed=sum(1 for a in user_acts if a.action_type == "material_view"),
                    videos_viewed=sum(1 for a in user_acts if a.action_type == "video_view"),
                    last_activity_at=max(a.created_at for a in user_acts)
                )
                db.add(progress)
            
            db.commit()
            
            # Re-query
            query = db.query(StudentProgress)
            if user_name:
                query = query.filter(StudentProgress.user_name == user_name)
            if subject_id:
                query = query.filter(StudentProgress.subject_id == subject_id)
            if group_id:
                query = query.filter(StudentProgress.group_id == group_id)
            
            progress_list = query.all()
    
    return progress_list


@router.get("/report", response_model=AnalyticsReport)
async def get_analytics_report(
    subject_id: Optional[UUID] = Query(None),
    group_id: Optional[UUID] = Query(None),
    user_name: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get aggregated analytics report"""
    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=days)
    
    # First, try to get data from StudentProgress
    progress_query = db.query(StudentProgress)
    if user_name:
        progress_query = progress_query.filter(StudentProgress.user_name == user_name)
    if subject_id:
        progress_query = progress_query.filter(StudentProgress.subject_id == subject_id)
    if group_id:
        progress_query = progress_query.filter(StudentProgress.group_id == group_id)
    
    progress_list = progress_query.all()
    
    # If we have progress data, use it
    if progress_list:
        # Calculate aggregates
        total_students = len(progress_list)
        total_tests = sum(p.tests_completed for p in progress_list)
        total_logins = sum(p.login_count for p in progress_list)
        total_time = sum(p.total_time_seconds for p in progress_list)
        
        # Calculate average score (weighted by number of tests)
        total_score_sum = 0.0
        total_tests_count = 0
        for p in progress_list:
            if p.average_score is not None and p.tests_completed > 0:
                total_score_sum += p.average_score * p.tests_completed
                total_tests_count += p.tests_completed
        
        avg_score = total_score_sum / total_tests_count if total_tests_count > 0 else 0.0
        
        # Calculate engagement score (0-100)
        max_possible_score = total_students * 100
        engagement_points = 0
        for p in progress_list:
            time_score = min(p.total_time_seconds / 3600, 100) * 0.4
            login_score = min(p.login_count * 2, 30)
            test_score = min((p.tests_completed / max(p.tests_total, 1)) * 30, 30)
            engagement_points += time_score + login_score + test_score
        
        engagement_score = (engagement_points / max_possible_score * 100) if max_possible_score > 0 else 0.0
        
        return AnalyticsReport(
            user_name=user_name,
            subject_id=subject_id,
            group_id=group_id,
            period_start=period_start,
            period_end=period_end,
            total_students=total_students,
            average_score=round(avg_score, 2),
            total_time_hours=round(total_time / 3600.0, 2),
            total_tests_completed=total_tests,
            total_logins=total_logins,
            engagement_score=round(engagement_score, 2)
        )
    
    # If no progress data, calculate from UserActivity
    activity_query = db.query(UserActivity).filter(
        UserActivity.created_at >= period_start,
        UserActivity.created_at <= period_end
    )
    
    if user_name:
        activity_query = activity_query.filter(UserActivity.user_name == user_name)
    
    activities = activity_query.all()
    
    if not activities:
        return AnalyticsReport(
            user_name=user_name,
            subject_id=subject_id,
            group_id=group_id,
            period_start=period_start,
            period_end=period_end,
            total_students=0,
            average_score=0.0,
            total_time_hours=0.0,
            total_tests_completed=0,
            total_logins=0,
            engagement_score=0.0
        )
    
    # Group activities by user
    user_stats = {}
    for activity in activities:
        if activity.user_name not in user_stats:
            user_stats[activity.user_name] = {
                "logins": 0,
                "test_actions": 0,
                "total_time": 0
            }
        
        if activity.action_type == "login":
            user_stats[activity.user_name]["logins"] += 1
        elif activity.action_type in ["test_start", "test_finish"]:
            user_stats[activity.user_name]["test_actions"] += 1
        
        if activity.session_duration:
            user_stats[activity.user_name]["total_time"] += activity.session_duration
    
    # Calculate aggregates from activities
    total_students = len(user_stats)
    total_logins = sum(stats["logins"] for stats in user_stats.values())
    total_tests = sum(stats["test_actions"] for stats in user_stats.values()) // 2  # Each test has start and finish
    total_time = sum(stats["total_time"] for stats in user_stats.values())
    
    # Calculate engagement score
    engagement_points = 0
    for stats in user_stats.values():
        time_score = min(stats["total_time"] / 3600, 100) * 0.4
        login_score = min(stats["logins"] * 2, 30)
        test_score = min((stats["test_actions"] / 2) * 5, 30)  # Approximate test completion
        engagement_points += time_score + login_score + test_score
    
    max_possible_score = total_students * 100
    engagement_score = (engagement_points / max_possible_score * 100) if max_possible_score > 0 else 0.0
    
    return AnalyticsReport(
        user_name=user_name,
        subject_id=subject_id,
        group_id=group_id,
        period_start=period_start,
        period_end=period_end,
        total_students=total_students,
        average_score=0.0,  # Can't calculate from activities alone
        total_time_hours=round(total_time / 3600.0, 2),
        total_tests_completed=total_tests,
        total_logins=total_logins,
        engagement_score=round(engagement_score, 2)
    )


@router.get("/activity-stats", response_model=List[ActivityStats])
async def get_activity_stats(
    user_name: str = Query(..., description="Username to get stats for"),
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get daily activity statistics for a user"""
    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=days)
    
    # Query activities grouped by date
    activities = db.query(UserActivity).filter(
        and_(
            UserActivity.user_name == user_name,
            UserActivity.created_at >= period_start,
            UserActivity.created_at <= period_end
        )
    ).all()
    
    # Group by date
    stats_by_date = {}
    for activity in activities:
        date_str = activity.created_at.date().isoformat()
        if date_str not in stats_by_date:
            stats_by_date[date_str] = {
                "total_time": 0,
                "login_count": 0,
                "test_actions": 0,
                "material_views": 0,
                "video_views": 0
            }
        
        if activity.action_type == "login":
            stats_by_date[date_str]["login_count"] += 1
        elif activity.action_type in ["test_start", "test_finish"]:
            stats_by_date[date_str]["test_actions"] += 1
        elif activity.action_type == "material_view":
            stats_by_date[date_str]["material_views"] += 1
        elif activity.action_type == "video_view":
            stats_by_date[date_str]["video_views"] += 1
        
        if activity.session_duration:
            stats_by_date[date_str]["total_time"] += activity.session_duration
    
    # Convert to list
    result = [
        ActivityStats(
            user_name=user_name,
            date=date_str,
            total_time_seconds=stats["total_time"],
            login_count=stats["login_count"],
            test_actions=stats["test_actions"],
            material_views=stats["material_views"],
            video_views=stats["video_views"]
        )
        for date_str, stats in sorted(stats_by_date.items())
    ]
    
    return result

