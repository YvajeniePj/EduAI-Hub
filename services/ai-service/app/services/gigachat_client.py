"""
GigaChat API client
"""
import os
import httpx
import logging
import base64
import uuid
from typing import Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY", "")
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

# Cache for access token
_access_token: Optional[str] = None
_token_expires_at: Optional[datetime] = None


async def get_access_token() -> Optional[str]:
    """
    Get GigaChat access token via OAuth
    Token is valid for 30 minutes
    """
    global _access_token, _token_expires_at
    
    # Return cached token if still valid (with 2 minute buffer)
    if _access_token and _token_expires_at and datetime.now() < (_token_expires_at - timedelta(minutes=2)):
        return _access_token
    
    if not GIGACHAT_API_KEY or GIGACHAT_API_KEY == "your_gigachat_api_key_here":
        logger.error("GIGACHAT_API_KEY not set or is placeholder. Please set a valid GigaChat Authorization key in .env file")
        return None
    
    try:
        # GIGACHAT_API_KEY should be the Authorization key from GigaChat
        # According to GigaChat docs, use Basic auth with format: base64(Authorization_key:)
        # Check if key is already base64 encoded (ends with = or ==)
        if GIGACHAT_API_KEY.endswith('=') or GIGACHAT_API_KEY.endswith('=='):
            # Key is already base64 encoded, use it directly
            auth_string = GIGACHAT_API_KEY
        elif ":" in GIGACHAT_API_KEY:
            # If key contains colon, it might be username:password format
            auth_string = base64.b64encode(GIGACHAT_API_KEY.encode()).decode()
        else:
            # Standard format: base64(Authorization_key:)
            auth_string = base64.b64encode(f"{GIGACHAT_API_KEY}:".encode()).decode()
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {auth_string}"
        }
        
        data = {
            "scope": "GIGACHAT_API_PERS"
        }
        
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:  # verify=False for self-signed cert
            response = await client.post(GIGACHAT_OAUTH_URL, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                _access_token = result.get("access_token")
                if _access_token:
                    # Token is valid for 30 minutes
                    _token_expires_at = datetime.now() + timedelta(minutes=30)
                    logger.info("GigaChat access token obtained successfully")
                    return _access_token
                else:
                    logger.error(f"GigaChat OAuth error: no access_token in response: {result}")
                    return None
            else:
                error_text = response.text
                try:
                    error_json = response.json()
                    logger.error(f"GigaChat OAuth error: {response.status_code} - {error_json}")
                except:
                    logger.error(f"GigaChat OAuth error: {response.status_code} - {error_text}")
                return None
    except Exception as e:
        logger.error(f"GigaChat OAuth request error: {e}")
        return None


async def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> Optional[str]:
    """
    Send a chat completion request to GigaChat
    Returns the response text or None on error
    """
    # Get access token
    access_token = await get_access_token()
    if not access_token:
        logger.error("Failed to get GigaChat access token")
        return None
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": "GigaChat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:  # verify=False for self-signed cert
            response = await client.post(GIGACHAT_API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"GigaChat API error: {response.status_code} - {response.text}")
                # If 401, clear token cache to force refresh
                if response.status_code == 401:
                    global _access_token, _token_expires_at
                    _access_token = None
                    _token_expires_at = None
                return None
    except Exception as e:
        logger.error(f"GigaChat request error: {e}")
        return None

