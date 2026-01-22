"""
Groups router - CRUD operations for groups
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID

from app.database import get_db
from app.models import Group, GroupMember, GroupRequest, Subject
from app.schemas import (
    GroupCreate, GroupResponse, GroupUpdate,
    GroupMemberCreate, GroupMemberResponse,
    GroupRequestCreate, GroupRequestResponse, GroupRequestUpdate
)

router = APIRouter()


@router.get("", response_model=List[GroupResponse])
async def get_groups(
    subject_id: Optional[UUID] = Query(None),
    user_name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all groups, optionally filtered by subject or user_name (member)"""
    query = db.query(Group)
    if subject_id:
        query = query.filter(Group.subject_id == subject_id)
    groups = query.all()
    
    # If user_name is provided, filter groups where user is a member
    if user_name:
        member_groups = db.query(GroupMember.group_id).filter(GroupMember.user_name == user_name).all()
        member_group_ids = {str(g[0]) for g in member_groups}
        groups = [g for g in groups if str(g.id) in member_group_ids]
    
    # Add member count
    result = []
    for group in groups:
        member_count = db.query(GroupMember).filter(GroupMember.group_id == group.id).count()
        group_dict = {
            "id": group.id,
            "subject_id": group.subject_id,
            "name": group.name,
            "description": group.description,
            "max_size": group.max_size,
            "created_by": group.created_by,
            "created_at": group.created_at,
            "member_count": member_count
        }
        result.append(group_dict)
    
    return result


@router.get("/{group_id}", response_model=GroupResponse)
async def get_group(group_id: UUID, db: Session = Depends(get_db)):
    """Get a specific group"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    member_count = db.query(GroupMember).filter(GroupMember.group_id == group.id).count()
    return {
        "id": group.id,
        "subject_id": group.subject_id,
        "name": group.name,
        "description": group.description,
        "created_at": group.created_at,
        "member_count": member_count
    }


@router.post("", response_model=GroupResponse, status_code=201)
async def create_group(group: GroupCreate, db: Session = Depends(get_db)):
    """Create a new group"""
    # Check if subject exists
    subject = db.query(Subject).filter(Subject.id == group.subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    # Check if group with same name in subject already exists
    existing = db.query(Group).filter(
        Group.subject_id == group.subject_id,
        Group.name == group.name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Group with this name already exists for this subject")
    
    db_group = Group(
        subject_id=group.subject_id,
        name=group.name,
        description=group.description,
        max_size=group.max_size,
        created_by=group.created_by
    )
    db.add(db_group)
    db.commit()
    db.refresh(db_group)
    
    return {
        "id": db_group.id,
        "subject_id": db_group.subject_id,
        "name": db_group.name,
        "description": db_group.description,
        "max_size": db_group.max_size,
        "created_by": db_group.created_by,
        "created_at": db_group.created_at,
        "member_count": 0
    }


@router.put("/{group_id}", response_model=GroupResponse)
async def update_group(group_id: UUID, group_update: GroupUpdate, db: Session = Depends(get_db)):
    """Update a group"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if group_update.name is not None:
        # Check if another group with this name exists in the same subject
        existing = db.query(Group).filter(
            Group.subject_id == group.subject_id,
            Group.name == group_update.name,
            Group.id != group_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Group with this name already exists for this subject")
        group.name = group_update.name
    
    if group_update.description is not None:
        group.description = group_update.description
    
    if group_update.max_size is not None:
        group.max_size = group_update.max_size
    
    db.commit()
    db.refresh(group)
    
    member_count = db.query(GroupMember).filter(GroupMember.group_id == group.id).count()
    return {
        "id": group.id,
        "subject_id": group.subject_id,
        "name": group.name,
        "description": group.description,
        "max_size": group.max_size,
        "created_by": group.created_by,
        "created_at": group.created_at,
        "member_count": member_count
    }


@router.delete("/{group_id}", status_code=200)
async def delete_group(group_id: UUID, db: Session = Depends(get_db)):
    """Delete a group"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    db.delete(group)
    db.commit()
    return {"message": "Group deleted successfully"}


@router.get("/{group_id}/members", response_model=List[GroupMemberResponse])
async def get_group_members(group_id: UUID, db: Session = Depends(get_db)):
    """Get all members of a group"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    members = db.query(GroupMember).filter(GroupMember.group_id == group_id).all()
    return members


@router.post("/{group_id}/members", response_model=GroupMemberResponse, status_code=201)
async def add_group_member(group_id: UUID, member: GroupMemberCreate, db: Session = Depends(get_db)):
    """Add a member to a group"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if member already exists in group
    existing = db.query(GroupMember).filter(
        GroupMember.group_id == group_id,
        GroupMember.user_name == member.user_name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="User is already a member of this group")
    
    db_member = GroupMember(
        group_id=group_id,
        user_name=member.user_name
    )
    db.add(db_member)
    db.commit()
    db.refresh(db_member)
    return db_member


@router.delete("/{group_id}/members/{member_id}", status_code=200)
async def remove_group_member(group_id: UUID, member_id: UUID, db: Session = Depends(get_db)):
    """Remove a member from a group"""
    member = db.query(GroupMember).filter(
        GroupMember.id == member_id,
        GroupMember.group_id == group_id
    ).first()
    if not member:
        raise HTTPException(status_code=404, detail="Group member not found")
    
    db.delete(member)
    db.commit()
    return {"message": "Member removed from group successfully"}


@router.delete("/{group_id}/members/by-user/{user_name}", status_code=200)
async def remove_group_member_by_user(group_id: UUID, user_name: str, db: Session = Depends(get_db)):
    """Remove a member from a group by username"""
    member = db.query(GroupMember).filter(
        GroupMember.group_id == group_id,
        GroupMember.user_name == user_name
    ).first()
    if not member:
        raise HTTPException(status_code=404, detail="Group member not found")
    
    db.delete(member)
    db.commit()
    return {"message": "Member removed from group successfully"}


# Group Requests endpoints
@router.post("/{group_id}/requests", response_model=GroupRequestResponse, status_code=201)
async def create_group_request(group_id: UUID, request: GroupRequestCreate, db: Session = Depends(get_db)):
    """Create a request to join a group"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Check if user is already a member
    existing_member = db.query(GroupMember).filter(
        GroupMember.group_id == group_id,
        GroupMember.user_name == request.user_name
    ).first()
    if existing_member:
        raise HTTPException(status_code=400, detail="User is already a member of this group")
    
    # Check if there's already a pending request
    existing_request = db.query(GroupRequest).filter(
        GroupRequest.group_id == group_id,
        GroupRequest.user_name == request.user_name,
        GroupRequest.status == "pending"
    ).first()
    if existing_request:
        raise HTTPException(status_code=400, detail="User already has a pending request for this group")
    
    # Check if group is full
    if group.max_size:
        member_count = db.query(GroupMember).filter(GroupMember.group_id == group_id).count()
        if member_count >= group.max_size:
            raise HTTPException(status_code=400, detail="Group is full")
    
    db_request = GroupRequest(
        group_id=group_id,
        user_name=request.user_name,
        status="pending"
    )
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    return db_request


@router.get("/{group_id}/requests", response_model=List[GroupRequestResponse])
async def get_group_requests(group_id: UUID, status: str = None, db: Session = Depends(get_db)):
    """Get all requests for a group, optionally filtered by status"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    query = db.query(GroupRequest).filter(GroupRequest.group_id == group_id)
    if status:
        query = query.filter(GroupRequest.status == status)
    
    requests = query.all()
    return requests


@router.put("/{group_id}/requests/{request_id}", response_model=GroupRequestResponse)
async def update_group_request(
    group_id: UUID, 
    request_id: UUID, 
    request_update: GroupRequestUpdate,
    db: Session = Depends(get_db)
):
    """Approve or reject a group request"""
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    db_request = db.query(GroupRequest).filter(
        GroupRequest.id == request_id,
        GroupRequest.group_id == group_id
    ).first()
    if not db_request:
        raise HTTPException(status_code=404, detail="Group request not found")
    
    if db_request.status != "pending":
        raise HTTPException(status_code=400, detail="Request has already been processed")
    
    from datetime import datetime
    db_request.status = request_update.status
    db_request.reviewed_at = datetime.utcnow()
    db_request.reviewed_by = request_update.reviewed_by
    
    # If approved, add user to group
    if request_update.status == "approved":
        # Check if group is full
        if group.max_size:
            member_count = db.query(GroupMember).filter(GroupMember.group_id == group_id).count()
            if member_count >= group.max_size:
                raise HTTPException(status_code=400, detail="Group is full")
        
        # Add member
        member = GroupMember(
            group_id=group_id,
            user_name=db_request.user_name
        )
        db.add(member)
    
    db.commit()
    db.refresh(db_request)
    return db_request


@router.get("/requests/my", response_model=List[GroupRequestResponse])
async def get_my_group_requests(user_name: str = Query(...), status: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """Get all requests made by a user"""
    query = db.query(GroupRequest).filter(GroupRequest.user_name == user_name)
    if status:
        query = query.filter(GroupRequest.status == status)
    
    requests = query.all()
    return requests

