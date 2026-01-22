"""
API Gateway - Entry point for all API requests
Routes requests to appropriate microservices
"""
from fastapi import FastAPI, HTTPException, Request, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import os
from typing import Optional
import logging
import jwt
from datetime import datetime, timedelta
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EduAI Hub API Gateway",
    description="API Gateway for EduAI Hub microservices",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
SUBJECT_SERVICE_URL = os.getenv("SUBJECT_SERVICE_URL", "http://subject-service:8001")
TEST_SERVICE_URL = os.getenv("TEST_SERVICE_URL", "http://test-service:8002")
SUBMISSION_SERVICE_URL = os.getenv("SUBMISSION_SERVICE_URL", "http://submission-service:8003")
MATERIAL_SERVICE_URL = os.getenv("MATERIAL_SERVICE_URL", "http://material-service:8004")
VIDEO_SERVICE_URL = os.getenv("VIDEO_SERVICE_URL", "http://video-service:8005")
PEER_REVIEW_SERVICE_URL = os.getenv("PEER_REVIEW_SERVICE_URL", "http://peer-review-service:8006")
GAMIFICATION_SERVICE_URL = os.getenv("GAMIFICATION_SERVICE_URL", "http://gamification-service:8007")
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://ai-service:8008")
ANALYTICS_SERVICE_URL = os.getenv("ANALYTICS_SERVICE_URL", "http://analytics-service:8009")
STREAMING_SERVICE_URL = os.getenv("STREAMING_SERVICE_URL", "http://streaming-service:8012")
NOTIFICATION_SERVICE_URL = os.getenv("NOTIFICATION_SERVICE_URL", "http://notification-service:8010")
FEEDBACK_SERVICE_URL = os.getenv("FEEDBACK_SERVICE_URL", "http://feedback-service:8011")

# HTTP client with timeout


# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "eduai_hub_secret_key_2024_change_in_production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

security = HTTPBearer(auto_error=False)


async def proxy_request(
    service_url: str,
    path: str,
    method: str = "GET",
    body: Optional[dict] = None,
    params: Optional[dict] = None,
    headers: Optional[dict] = None
):
    """Proxy request to a microservice"""
    url = f"{service_url}{path}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                response = await client.get(url, params=params, headers=headers)
            elif method == "POST":
                response = await client.post(url, json=body, params=params, headers=headers)
            elif method == "PUT":
                response = await client.put(url, json=body, params=params, headers=headers)
            elif method == "DELETE":
                response = await client.delete(url, params=params, headers=headers)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")
            
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                return None, response.status_code, error_detail
            
            try:
                return response.json(), response.status_code, None
            except:
                return response.text, response.status_code, None
    except httpx.RequestError as e:
        logger.error(f"Request error to {url}: {e}")
        # Log resolution info for debugging
        try:
            import socket
            hostname = service_url.split("//")[-1].split(":")[0]
            ip = socket.gethostbyname(hostname)
            logger.info(f"Resolved {hostname} to {ip}")
        except:
            pass
        raise HTTPException(status_code=503, detail=f"Service unavailable: {service_url}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def create_jwt_token(user_id: str, username: str, role: str = "student") -> str:
    """Create JWT token for user"""
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify JWT token and return user data"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {
            "user_id": payload.get("user_id"),
            "username": payload.get("username"),
            "role": payload.get("role", "student")
        }
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except Exception:
        return None

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    """Get current user from JWT token"""
    if not credentials:
        return None
    return verify_jwt_token(credentials.credentials)


async def get_current_teacher(current_user: dict = Depends(get_current_user)):
    """Dependency to check if user is a teacher"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if current_user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="Not authorized. Teacher role required.")
    return current_user


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "api-gateway"}


# Auth endpoints
@app.post("/auth/login")
async def login(request: Request):
    """Login user and return JWT token"""
    body = await request.json()
    username = body.get("name") or body.get("username")
    
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    
    # Get user from submission service
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, f"/users/by-name/{username}", "GET")
    if status == 404:
        raise HTTPException(status_code=404, detail="User not found")
    elif status != 200:
        # If service is unavailable (502, 503, etc.), return a more helpful error
        logger.error(f"Submission service error during login: {status}, {error}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service unavailable. Please try again later. (Error: {error or 'Unknown error'})"
        )
    
    user = data
    role = user.get("role", "student")
    # Create JWT token
    token = create_jwt_token(str(user["id"]), user["name"], role)
    
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "avatar_url": user.get("avatar_url"),
            "role": role
        }
    }


@app.post("/auth/register")
async def register(request: Request):
    """Register new user and return JWT token"""
    body = await request.json()
    username = body.get("name") or body.get("username")
    
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    
    # First check if user already exists
    check_data, check_status, check_error = await proxy_request(SUBMISSION_SERVICE_URL, f"/users/by-name/{username}", "GET")
    if check_status == 200:
        # User already exists
        raise HTTPException(status_code=409, detail="Username already taken")
    elif check_status not in [404, 503, 502]:
        # Some other error (not "not found" and not service unavailable)
        logger.error(f"Unexpected error checking user existence: {check_status}, {check_error}")
        raise HTTPException(status_code=check_status, detail=check_error or "Failed to check user existence")
    
    # If we got 404, user doesn't exist, proceed with creation
    # If we got 502/503, service is down, but we'll try to create anyway (might be transient)
    
    # Create user in submission service
    role = body.get("role", "student")
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, "/users", "POST", {"name": username, "role": role})
    if status == 409 or (isinstance(error, dict) and error.get("detail") and "already exists" in str(error.get("detail")).lower()):
        raise HTTPException(status_code=409, detail="Username already taken")
    elif status not in [200, 201]:
        logger.error(f"Submission service error during registration: {status}, {error}")
        if status in [502, 503]:
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable. Please try again later. (Error: {error or 'Unknown error'})"
            )
        raise HTTPException(status_code=status, detail=error or "Failed to create user")
    
    user = data
    # Create JWT token
    token = create_jwt_token(str(user["id"]), user["name"], role)
    
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "avatar_url": user.get("avatar_url"),
            "role": role
        }
    }


@app.get("/auth/me")
async def get_current_user_info(current_user: Optional[dict] = Depends(get_current_user)):
    """Get current user info from JWT token"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return current_user


# Subject Service Routes
@app.get("/subjects")
async def get_subjects():
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, "/subjects", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch subjects")
    return data


@app.post("/subjects")
async def create_subject(request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    # URL encode the username to handle Cyrillic characters safely in headers
    encoded_name = quote(current_user["username"])
    headers = {"X-User-Name": encoded_name}
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, "/subjects", "POST", body, headers=headers)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create subject")
    return data


@app.delete("/subjects/{subject_id}")
async def delete_subject(subject_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/subjects/{subject_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete subject")
    return data


@app.put("/subjects/{subject_id}")
async def update_subject(subject_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/subjects/{subject_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update subject")
    return data


@app.post("/subjects/{subject_id}/cover")
async def upload_subject_cover(subject_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    """Upload a cover image for a subject"""
    form = await request.form()
    
    files = {}
    for key, value in form.items():
        if hasattr(value, 'filename') and hasattr(value, 'read'):
            file_content = await value.read()
            files[key] = (value.filename, file_content, value.content_type or "image/jpeg")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{SUBJECT_SERVICE_URL}/subjects/{subject_id}/cover",
                files=files
            )
            
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Request error to subject service: {e}")
        raise HTTPException(status_code=503, detail="Subject service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/subjects/{subject_id}/cover")
async def get_subject_cover(subject_id: str):
    """Get the cover image for a subject"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{SUBJECT_SERVICE_URL}/subjects/{subject_id}/cover")
            
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail="Cover not found")
            
            from fastapi.responses import Response
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "image/jpeg")
            )
    except httpx.RequestError as e:
        logger.error(f"Request error to subject service: {e}")
        raise HTTPException(status_code=503, detail="Subject service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Course Structure Routes
@app.get("/subjects/{subject_id}/structure")
async def get_course_structure(subject_id: str):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/subjects/{subject_id}/structure", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch course structure")
    return data


@app.post("/subjects/{subject_id}/modules")
async def create_module(subject_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/subjects/{subject_id}/modules", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create module")
    return data


@app.put("/modules/{module_id}")
async def update_module(module_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/modules/{module_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update module")
    return data


@app.delete("/modules/{module_id}")
async def delete_module(module_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/modules/{module_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete module")
    return data


@app.post("/modules/{module_id}/lessons")
async def create_lesson(module_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/modules/{module_id}/lessons", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create lesson")
    return data


@app.put("/lessons/{lesson_id}")
async def update_lesson(lesson_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/lessons/{lesson_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update lesson")
    return data


@app.delete("/lessons/{lesson_id}")
async def delete_lesson(lesson_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/lessons/{lesson_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete lesson")
    return data


@app.post("/lessons/{lesson_id}/content")
async def create_content(lesson_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/lessons/{lesson_id}/content", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create content")
    return data


@app.put("/content/{content_id}")
async def update_content(content_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/content/{content_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update content")
    return data


@app.get("/lessons/{lesson_id}/content")
async def get_content(lesson_id: str):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/lessons/{lesson_id}/content", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch content")
    return data


# News Routes
@app.get("/news")
async def get_news(subject_id: Optional[str] = None):
    params = {"subject_id": subject_id} if subject_id else None
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, "/news", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch news")
    return data


@app.get("/news/{news_id}")
async def get_news_item(news_id: str):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/news/{news_id}", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch news")
    return data


@app.post("/news")
async def create_news(request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, "/news", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create news")
    return data


@app.put("/news/{news_id}")
async def update_news(news_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/news/{news_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update news")
    return data


@app.delete("/news/{news_id}")
async def delete_news(news_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/news/{news_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete news")
    return data


# Test Service Routes
@app.get("/tests")
async def get_tests(subject_id: Optional[str] = None):
    params = {"subject_id": subject_id} if subject_id else None
    data, status, error = await proxy_request(TEST_SERVICE_URL, "/tests", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch tests")
    return data


@app.get("/tests/{test_id}")
async def get_test(test_id: str):
    data, status, error = await proxy_request(TEST_SERVICE_URL, f"/tests/{test_id}", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch test")
    return data


@app.post("/tests")
async def create_test(request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    # URL encode the username to handle Cyrillic characters safely in headers
    encoded_name = quote(current_user["username"])
    headers = {"X-User-Name": encoded_name}
    data, status, error = await proxy_request(TEST_SERVICE_URL, "/tests", "POST", body, headers=headers)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create test")
    return data


@app.put("/tests/{test_id}")
async def update_test(test_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(TEST_SERVICE_URL, f"/tests/{test_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update test")
    return data


@app.delete("/tests/{test_id}")
async def delete_test(test_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(TEST_SERVICE_URL, f"/tests/{test_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete test")
    return data


# Submission Service Routes
@app.get("/submissions")
async def get_submissions(test_id: Optional[str] = None, user: Optional[str] = None):
    params = {}
    if test_id:
        params["test_id"] = test_id
    if user:
        params["user"] = user
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, "/submissions", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch submissions")
    return data


@app.post("/submissions")
async def create_submission(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, "/submissions", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create submission")
    return data


@app.put("/submissions/{submission_id}")
async def update_submission(submission_id: str, request: Request):
    body = await request.json()
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, f"/submissions/{submission_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update submission")
    return data


@app.post("/submissions/{submission_id}/finish")
async def finish_submission(submission_id: str, use_ai: bool = False):
    params = {"use_ai": use_ai} if use_ai else None
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, f"/submissions/{submission_id}/finish", "POST", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to finish submission")
    return data


@app.get("/submissions/{submission_id}/results")
async def get_submission_results(submission_id: str):
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, f"/submissions/{submission_id}/results", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch results")
    return data

# Users (handled in submission-service)
@app.get("/users")
async def get_users():
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, "/users", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch users")
    return data


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, f"/users/{user_id}", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch user")
    return data


@app.get("/users/by-name/{name}")
async def get_user_by_name(name: str):
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, f"/users/by-name/{name}", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch user")
    return data


@app.post("/users")
async def create_user(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, "/users", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create user")
    return data


@app.put("/users/{user_id}")
async def update_user(user_id: str, request: Request, current_user: Optional[dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    body = await request.json()
    data, status, error = await proxy_request(SUBMISSION_SERVICE_URL, f"/users/{user_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update user")
    return data


@app.post("/users/{user_id}/avatar")
async def upload_avatar(user_id: str, request: Request, current_user: Optional[dict] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    form = await request.form()
    files = {}
    for key, value in form.items():
        if hasattr(value, 'filename') and hasattr(value, 'read'):
            file_content = await value.read()
            files[key] = (value.filename, file_content, value.content_type or "image/jpeg")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client_async:
            response = await client_async.post(
                f"{SUBMISSION_SERVICE_URL}/users/{user_id}/avatar",
                files=files
            )
            
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Request error to submission service: {e}")
        raise HTTPException(status_code=503, detail="Submission service unavailable")


# Material Service Routes
@app.get("/materials")
async def get_materials(subject_id: Optional[str] = None):
    params = {"subject_id": subject_id} if subject_id else None
    data, status, error = await proxy_request(MATERIAL_SERVICE_URL, "/materials", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch materials")
    return data


@app.post("/materials")
async def create_material(request: Request, current_user: dict = Depends(get_current_teacher)):
    """Upload a material file - proxy multipart form data"""
    form = await request.form()
    
    # Forward multipart form data to material service
    files = {}
    data = {}
    for key, value in form.items():
        # Check if it's a file (UploadFile)
        if hasattr(value, 'filename') and hasattr(value, 'read'):
            # It's a file
            file_content = await value.read()
            files[key] = (value.filename, file_content, value.content_type or "application/octet-stream")
        else:
            # It's regular form data
            data[key] = value
    
    # URL encode the username to handle Cyrillic characters safely in headers
    encoded_name = quote(current_user["username"])
    headers = {"X-User-Name": encoded_name}
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes for large files
            response = await client.post(
                f"{MATERIAL_SERVICE_URL}/materials",
                files=files if files else None,
                data=data if data else None,
                headers=headers
            )
            
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Request error to material service: {e}")
        raise HTTPException(status_code=503, detail="Material service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/materials/{material_id}")
async def get_material(material_id: str):
    data, status, error = await proxy_request(MATERIAL_SERVICE_URL, f"/materials/{material_id}", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch material")
    return data


@app.get("/materials/{material_id}/text")
async def get_material_text(material_id: str):
    data, status, error = await proxy_request(MATERIAL_SERVICE_URL, f"/materials/{material_id}/text", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch material text")
    return data


@app.post("/materials/{material_id}/annotate")
async def create_material_annotation(material_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(MATERIAL_SERVICE_URL, f"/materials/{material_id}/annotate", "POST")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to create annotation")
    return data


@app.delete("/materials/{material_id}")
async def delete_material(material_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(MATERIAL_SERVICE_URL, f"/materials/{material_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete material")
    return data


@app.get("/materials/{material_id}/download")
async def download_material(material_id: str):
    """Download a material file - proxy the file response"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"{MATERIAL_SERVICE_URL}/materials/{material_id}/download")
            
            if response.status_code >= 400:
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            # Return streaming response with proper headers
            from fastapi.responses import Response
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "application/octet-stream"),
                headers={
                    "Content-Disposition": response.headers.get("content-disposition", f"attachment; filename=material_{material_id}")
                }
            )
    except httpx.RequestError as e:
        logger.error(f"Request error to material service: {e}")
        raise HTTPException(status_code=503, detail="Material service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Video Service Routes
@app.get("/videos")
async def get_videos(subject_id: Optional[str] = None):
    params = {"subject_id": subject_id} if subject_id else None
    data, status, error = await proxy_request(VIDEO_SERVICE_URL, "/videos", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch videos")
    return data


@app.post("/videos")
async def create_video(request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    # URL encode the username to handle Cyrillic characters safely in headers
    encoded_name = quote(current_user["username"])
    headers = {"X-User-Name": encoded_name}
    data, status, error = await proxy_request(VIDEO_SERVICE_URL, "/videos", "POST", body, headers=headers)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create video")
    return data


@app.delete("/videos/{video_id}")
async def delete_video(video_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(VIDEO_SERVICE_URL, f"/videos/{video_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete video")
    return data


# AI Service Routes
@app.get("/ai/status")
async def get_ai_status():
    data, status, error = await proxy_request(AI_SERVICE_URL, "/ai/status", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to get AI status")
    return data


@app.post("/ai/annotate")
async def annotate_material(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(AI_SERVICE_URL, "/ai/annotate", "POST", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to create annotation")
    return data


@app.post("/ai/grade")
async def grade_answer(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(AI_SERVICE_URL, "/ai/grade", "POST", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to grade answer")
    return data


@app.post("/ai/chat")
async def chat_assistant(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(AI_SERVICE_URL, "/ai/chat", "POST", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to process chat")
    return data


@app.post("/ai/generate-test")
async def generate_test(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(AI_SERVICE_URL, "/ai/generate-test", "POST", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to generate test")
    return data


@app.post("/ai/generate-course")
async def generate_course(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(AI_SERVICE_URL, "/ai/generate-course", "POST", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to generate course")
    return data


@app.post("/ai/test-feedback")
async def get_test_feedback(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(AI_SERVICE_URL, "/ai/test-feedback", "POST", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to get test feedback")
    return data


# Peer Review Service Routes
@app.get("/reviews")
async def get_reviews(submission_id: Optional[str] = None):
    params = {"submission_id": submission_id} if submission_id else None
    data, status, error = await proxy_request(PEER_REVIEW_SERVICE_URL, "/reviews", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch reviews")
    return data


@app.get("/reviews/submissions-for-review")
async def get_submissions_for_review(test_id: str, reviewer: str):
    params = {"test_id": test_id, "reviewer": reviewer}
    data, status, error = await proxy_request(PEER_REVIEW_SERVICE_URL, "/reviews/submissions-for-review", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch submissions for review")
    return data


@app.get("/reviews/my-reviews")
async def get_my_reviews(user: str, test_id: Optional[str] = None):
    params = {"user": user}
    if test_id:
        params["test_id"] = test_id
    data, status, error = await proxy_request(PEER_REVIEW_SERVICE_URL, "/reviews/my-reviews", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch my reviews")
    return data


@app.post("/reviews")
async def create_review(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(PEER_REVIEW_SERVICE_URL, "/reviews", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create review")
    return data


# Gamification Service Routes
@app.get("/points")
async def get_leaderboard(subject_id: Optional[str] = None, limit: Optional[int] = 100):
    params = {"limit": limit}
    if subject_id:
        params["subject_id"] = subject_id
    data, status, error = await proxy_request(GAMIFICATION_SERVICE_URL, "/points", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch leaderboard")
    return data


@app.get("/points/export")
async def export_leaderboard(subject_id: Optional[str] = None):
    params = {"subject_id": subject_id} if subject_id else None
    data, status, error = await proxy_request(GAMIFICATION_SERVICE_URL, "/points/export", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to export leaderboard")
    return data


@app.get("/points/{username}")
async def get_user_points(username: str, subject_id: Optional[str] = None):
    params = {"subject_id": subject_id} if subject_id else None
    data, status, error = await proxy_request(GAMIFICATION_SERVICE_URL, f"/points/{username}", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch user points")
    return data


@app.post("/points")
async def award_points(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(GAMIFICATION_SERVICE_URL, "/points", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to award points")
    return data


# Analytics Service Routes
ANALYTICS_SERVICE_URL = os.getenv("ANALYTICS_SERVICE_URL", "http://analytics-service:8009")

@app.post("/analytics/activities")
async def create_activity(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/activities", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create activity")
    return data


@app.get("/analytics/report")
async def get_analytics_report(
    subject_id: Optional[str] = None,
    group_id: Optional[str] = None,
    user_name: Optional[str] = None,
    days: int = 30
):
    params = {"days": days}
    if subject_id:
        params["subject_id"] = subject_id
    if group_id:
        params["group_id"] = group_id
    if user_name:
        params["user_name"] = user_name
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/report", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch analytics report")
    return data


@app.get("/analytics/progress")
async def get_analytics_progress(
    user_name: Optional[str] = None,
    subject_id: Optional[str] = None,
    group_id: Optional[str] = None
):
    params = {}
    if user_name:
        params["user_name"] = user_name
    if subject_id:
        params["subject_id"] = subject_id
    if group_id:
        params["group_id"] = group_id
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/progress", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch progress")
    return data


# --- Streaming Service ---

@app.post("/streaming/rooms/create")
async def create_streaming_room(room_data: dict):
    data, status, error = await proxy_request(STREAMING_SERVICE_URL, "/streaming/rooms/create", "POST", room_data)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create streaming room")
    return data

@app.post("/streaming/tokens/generate")
async def generate_streaming_token(request: dict):
    data, status, error = await proxy_request(STREAMING_SERVICE_URL, "/streaming/tokens/generate", "POST", request)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to generate streaming token")
    return data

@app.get("/streaming/rooms/active")
async def get_active_streaming_rooms():
    data, status, error = await proxy_request(STREAMING_SERVICE_URL, "/streaming/rooms/active")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch active streaming rooms")
    return data

@app.post("/streaming/rooms/{room_name}/end")
async def end_streaming_room(room_name: str):
    data, status, error = await proxy_request(STREAMING_SERVICE_URL, f"/streaming/rooms/{room_name}/end", "POST")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to end streaming room")
    return data


# Groups Routes (Subject Service)
@app.get("/groups")
async def get_groups(subject_id: Optional[str] = None, user_name: Optional[str] = None):
    params = {}
    if subject_id:
        params["subject_id"] = subject_id
    if user_name:
        params["user_name"] = user_name
        
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, "/groups", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch groups")
    return data


@app.post("/groups")
async def create_group(request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, "/groups", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create group")
    return data


@app.get("/groups/{group_id}")
async def get_group(group_id: str):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch group")
    return data


@app.put("/groups/{group_id}")
async def update_group(group_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update group")
    return data


@app.delete("/groups/{group_id}")
async def delete_group(group_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete group")
    return data


@app.get("/groups/{group_id}/members")
async def get_group_members(group_id: str):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}/members", "GET")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch group members")
    return data


@app.post("/groups/{group_id}/members")
async def add_group_member(group_id: str, request: Request, current_user: dict = Depends(get_current_teacher)):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}/members", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to add group member")
    return data


@app.delete("/groups/{group_id}/members/{member_id}")
async def remove_group_member(group_id: str, member_id: str, current_user: dict = Depends(get_current_teacher)):
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}/members/{member_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to remove group member")
    return data


# Group Requests Routes
@app.post("/groups/{group_id}/requests")
async def create_group_request(group_id: str, request: Request):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}/requests", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create group request")
    return data


@app.get("/groups/{group_id}/requests")
async def get_group_requests(group_id: str, status: Optional[str] = None):
    params = {}
    if status:
        params["status"] = status
    data, status_code, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}/requests", "GET", params=params)
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=error or "Failed to fetch group requests")
    return data


@app.put("/groups/{group_id}/requests/{request_id}")
async def update_group_request(group_id: str, request_id: str, request: Request):
    body = await request.json()
    data, status, error = await proxy_request(SUBJECT_SERVICE_URL, f"/groups/{group_id}/requests/{request_id}", "PUT", body)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to update group request")
    return data


@app.get("/groups/requests/my")
async def get_my_group_requests(user_name: str, status: Optional[str] = None):
    params = {"user_name": user_name}
    if status:
        params["status"] = status
    data, status_code, error = await proxy_request(SUBJECT_SERVICE_URL, "/groups/requests/my", "GET", params=params)
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=error or "Failed to fetch my group requests")
    return data


# Notification Service Routes
@app.get("/notifications")
async def get_notifications(user_name: Optional[str] = None, is_read: Optional[bool] = None):
    params = {}
    if user_name:
        params["user_name"] = user_name
    if is_read is not None:
        params["is_read"] = is_read
    data, status, error = await proxy_request(NOTIFICATION_SERVICE_URL, "/notifications", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch notifications")
    return data


@app.get("/notifications/count")
async def get_notification_count(user_name: str):
    params = {"user_name": user_name}
    data, status, error = await proxy_request(NOTIFICATION_SERVICE_URL, "/notifications/count", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to get notification count")
    return data


@app.post("/notifications")
async def create_notification(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(NOTIFICATION_SERVICE_URL, "/notifications", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create notification")
    return data


@app.post("/notifications/{notification_id}/mark-read")
async def mark_notification_read(notification_id: str):
    data, status, error = await proxy_request(NOTIFICATION_SERVICE_URL, f"/notifications/{notification_id}/mark-read", "POST")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to mark notification as read")
    return data


@app.post("/notifications/mark-all-read")
async def mark_all_notifications_read(user_name: str):
    params = {"user_name": user_name}
    data, status, error = await proxy_request(NOTIFICATION_SERVICE_URL, "/notifications/mark-all-read", "POST", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to mark all notifications as read")
    return data


@app.delete("/notifications/{notification_id}")
async def delete_notification(notification_id: str):
    data, status, error = await proxy_request(NOTIFICATION_SERVICE_URL, f"/notifications/{notification_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete notification")
    return data


# Feedback Service Routes
@app.get("/feedbacks")
async def get_feedbacks(user_name: Optional[str] = None, subject_id: Optional[str] = None, group_id: Optional[str] = None):
    params = {}
    if user_name:
        params["user_name"] = user_name
    if subject_id:
        params["subject_id"] = subject_id
    if group_id:
        params["group_id"] = group_id
    data, status, error = await proxy_request(FEEDBACK_SERVICE_URL, "/feedbacks", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch feedbacks")
    return data


@app.get("/feedbacks/stats")
async def get_feedback_stats(subject_id: Optional[str] = None, group_id: Optional[str] = None):
    params = {}
    if subject_id:
        params["subject_id"] = subject_id
    if group_id:
        params["group_id"] = group_id
    data, status, error = await proxy_request(FEEDBACK_SERVICE_URL, "/feedbacks/stats", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch feedback stats")
    return data


@app.post("/feedbacks")
async def create_feedback(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(FEEDBACK_SERVICE_URL, "/feedbacks", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create feedback")
    return data


@app.delete("/feedbacks/{feedback_id}")
async def delete_feedback(feedback_id: str):
    data, status, error = await proxy_request(FEEDBACK_SERVICE_URL, f"/feedbacks/{feedback_id}", "DELETE")
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to delete feedback")
    return data


# Analytics Service Routes
@app.post("/analytics/activities")
async def create_activity(request: Request):
    body = await request.json()
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/activities", "POST", body)
    if status not in [200, 201]:
        raise HTTPException(status_code=status, detail=error or "Failed to create activity")
    return data


@app.get("/analytics/activities")
async def get_activities(user_name: Optional[str] = None, action_type: Optional[str] = None):
    params = {}
    if user_name:
        params["user_name"] = user_name
    if action_type:
        params["action_type"] = action_type
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/activities", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch activities")
    return data


@app.get("/analytics/progress")
async def get_progress(user_name: Optional[str] = None, subject_id: Optional[str] = None, group_id: Optional[str] = None):
    params = {}
    if user_name:
        params["user_name"] = user_name
    if subject_id:
        params["subject_id"] = subject_id
    if group_id:
        params["group_id"] = group_id
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/progress", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch progress")
    return data


@app.get("/analytics/report")
async def get_analytics_report(
    subject_id: Optional[str] = None, 
    group_id: Optional[str] = None, 
    user_name: Optional[str] = None, 
    days: int = 30,
    current_user: dict = Depends(get_current_teacher)
):
    params = {"days": days}
    if subject_id:
        params["subject_id"] = subject_id
    if group_id:
        params["group_id"] = group_id
    if user_name:
        params["user_name"] = user_name
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/report", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch analytics report")
    return data


@app.get("/analytics/activity-stats")
async def get_activity_stats(user_name: str, days: int = 30):
    params = {"user_name": user_name, "days": days}
    data, status, error = await proxy_request(ANALYTICS_SERVICE_URL, "/analytics/activity-stats", "GET", params=params)
    if status != 200:
        raise HTTPException(status_code=status, detail=error or "Failed to fetch activity stats")
    return data


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

