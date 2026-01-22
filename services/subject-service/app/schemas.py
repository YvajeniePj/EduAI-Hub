"""
Pydantic schemas for Subject Service
"""
from pydantic import BaseModel, field_validator
from datetime import datetime, timezone, timedelta
from uuid import UUID
from typing import Optional


class SubjectBase(BaseModel):
    name: str
    description: Optional[str] = None


class SubjectCreate(SubjectBase):
    pass


class SubjectResponse(SubjectBase):
    id: UUID
    cover_image: Optional[str] = None
    rating: Optional[float] = None
    rating_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# News schemas
class NewsBase(BaseModel):
    subject_id: UUID
    title: str
    content: str
    image_url: Optional[str] = None


class NewsCreate(NewsBase):
    pass


class NewsUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    image_url: Optional[str] = None
    subject_id: Optional[UUID] = None


class NewsResponse(NewsBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def add_moscow_timezone(cls, v):
        """Добавляем московский часовой пояс (+03:00) к датам при возврате"""
        if v is None:
            return v
        if isinstance(v, datetime):
            moscow_tz = timezone(timedelta(hours=3))
            if v.tzinfo is None:
                # Если timezone нет (UTC из базы), предполагаем что это UTC и конвертируем в московское
                return v.replace(tzinfo=timezone.utc).astimezone(moscow_tz)
            else:
                # Если timezone есть, конвертируем в московское
                return v.astimezone(moscow_tz)
        return v

    class Config:
        from_attributes = True


# Group schemas
class GroupBase(BaseModel):
    subject_id: UUID
    name: str
    description: Optional[str] = None
    max_size: Optional[int] = None
    created_by: Optional[str] = None


class GroupCreate(GroupBase):
    pass


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    max_size: Optional[int] = None


class GroupResponse(GroupBase):
    id: UUID
    created_at: datetime
    member_count: Optional[int] = None

    class Config:
        from_attributes = True


# GroupMember schemas
class GroupMemberBase(BaseModel):
    group_id: UUID
    user_name: str


class GroupMemberCreate(GroupMemberBase):
    pass


class GroupMemberResponse(GroupMemberBase):
    id: UUID
    joined_at: datetime

    class Config:
        from_attributes = True


# GroupRequest schemas
class GroupRequestBase(BaseModel):
    group_id: UUID
    user_name: str


class GroupRequestCreate(BaseModel):
    user_name: str


class GroupRequestResponse(GroupRequestBase):
    id: UUID
    status: str
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None

    class Config:
        from_attributes = True


class GroupRequestUpdate(BaseModel):
    status: str  # approved or rejected
    reviewed_by: str


# Course structure schemas



class CourseLessonBase(BaseModel):
    title: str
    lesson_type: str  # lecture, quiz, video, material, exercise


class CourseLessonCreate(CourseLessonBase):
    pass


class CourseLessonUpdate(BaseModel):
    title: Optional[str] = None
    lesson_type: Optional[str] = None
    order_index: Optional[int] = None


class CourseContentBase(BaseModel):
    lesson_id: UUID
    text_content: Optional[str] = None
    video_url: Optional[str] = None
    video_platform: Optional[str] = None
    material_id: Optional[UUID] = None
    test_id: Optional[UUID] = None
    extra_data: Optional[dict] = None


class CourseContentCreate(CourseContentBase):
    pass


class CourseContentUpdate(BaseModel):
    text_content: Optional[str] = None
    video_url: Optional[str] = None
    video_platform: Optional[str] = None
    material_id: Optional[UUID] = None
    test_id: Optional[UUID] = None
    extra_data: Optional[dict] = None


class CourseContentResponse(CourseContentBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CourseLessonResponse(CourseLessonBase):
    id: UUID
    module_id: UUID
    order_index: int
    created_at: datetime
    updated_at: datetime
    content: Optional[CourseContentResponse] = None

    class Config:
        from_attributes = True



class CourseModuleBase(BaseModel):
    title: str
    description: Optional[str] = None
    is_collapsed: bool = False


class CourseModuleCreate(CourseModuleBase):
    pass


class CourseModuleUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    order_index: Optional[int] = None
    is_collapsed: Optional[bool] = None


class CourseModuleResponse(CourseModuleBase):
    id: UUID
    subject_id: UUID
    order_index: int
    created_at: datetime
    updated_at: datetime
    lessons: Optional[list[CourseLessonResponse]] = None

    class Config:
        from_attributes = True
class CourseStructureResponse(BaseModel):
    """Full course structure with modules, lessons, and content"""
    subject: SubjectResponse
    modules: list[CourseModuleResponse]
