"""
AI router - AI functions endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import re
import os
import httpx

from app.services.gigachat_client import chat_completion, get_access_token
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Service URLs
MATERIAL_SERVICE_URL = os.getenv("MATERIAL_SERVICE_URL", "http://material-service:8004")
VIDEO_SERVICE_URL = os.getenv("VIDEO_SERVICE_URL", "http://video-service:8005")
TEST_SERVICE_URL = os.getenv("TEST_SERVICE_URL", "http://test-service:8002")
SUBMISSION_SERVICE_URL = os.getenv("SUBMISSION_SERVICE_URL", "http://submission-service:8003")
SUBJECT_SERVICE_URL = os.getenv("SUBJECT_SERVICE_URL", "http://subject-service:8001")


class AnnotateRequest(BaseModel):
    text: str
    filename: str
    language: Optional[str] = "ru"  # "ru" for Russian, "en" for English


class GradeRequest(BaseModel):
    answer_text: str
    rubric: Dict


class GradeResponse(BaseModel):
    score: int
    feedback: List[str]


@router.get("/status")
async def get_ai_status():
    """Check if AI service (GigaChat) is available"""
    try:
        # Try to get access token to verify connection
        token = await get_access_token()
        if token:
            return {
                "status": "connected",
                "available": True,
                "message": "GigaChat подключен"
            }
        else:
            return {
                "status": "disconnected",
                "available": False,
                "message": "GigaChat не подключен"
            }
    except Exception as e:
        logger.error(f"Error checking AI status: {e}")
        return {
            "status": "error",
            "available": False,
            "message": f"Ошибка проверки статуса: {str(e)}"
        }


@router.post("/annotate")
async def annotate_material(request: AnnotateRequest):
    """Create an annotation for a material"""
    # Determine language for prompt
    lang = request.language.lower() if request.language else "ru"
    
    if lang == "en":
        system_msg = (
            "You are an educational assistant. Create a brief annotation (2-3 sentences) "
            "for the provided educational material. Focus on key concepts and practical value. "
            "Respond in English."
        )
        user_msg = f"Material: {request.filename}\n\nText:\n{request.text}"
    else:  # Default to Russian
        system_msg = (
            "Ты - образовательный ассистент. Создай краткую аннотацию (2-3 предложения) "
            "для предоставленного учебного материала. Сосредоточься на ключевых концепциях и практической ценности. "
            "Отвечай на русском языке."
        )
        user_msg = f"Материал: {request.filename}\n\nТекст:\n{request.text}"
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    result = await chat_completion(messages, temperature=0.3, max_tokens=200)
    if result:
        return {"annotation": result}
    else:
        raise HTTPException(status_code=503, detail="AI service unavailable")


@router.post("/grade", response_model=GradeResponse)
async def grade_answer(request: GradeRequest):
    """Grade an answer using AI"""
    system_msg = (
        "You are a strict but fair TA. "
        "Grade the student's short answer on a 0–100 scale based ONLY on the rubric. "
        "Return strict JSON: {\"score\": <int>, \"feedback\": [\"...\", \"...\"]}"
    )
    
    keywords = ", ".join([k.get("word", "") for k in request.rubric.get("keywords", [])])
    user_msg = (
        f"Rubric title: {request.rubric.get('title', '')}\n"
        f"Max points: 100\n"
        f"Keywords (hints): {keywords}\n"
        f"Student answer:\n{request.answer_text}"
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    result = await chat_completion(messages, temperature=0.2, max_tokens=500)
    if result:
        # Extract JSON from response
        try:
            m = re.search(r"\{.*\}", result, re.S)
            json_str = m.group(0) if m else result
            data = json.loads(json_str)
            score = int(max(0, min(100, data.get("score", 0))))
            feedback = data.get("feedback", [])
            if not isinstance(feedback, list):
                feedback = [str(feedback)]
            return GradeResponse(score=score, feedback=feedback[:5])
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse AI response")
    else:
        raise HTTPException(status_code=503, detail="AI service unavailable")


class ChatRequest(BaseModel):
    question: str
    subject_id: Optional[str] = None


@router.post("/chat")
async def chat_assistant(request: ChatRequest):
    """Chat assistant with context from materials, videos, and tests"""
    try:
        # Build context from various services
        context_parts = []
        question_lower = request.question.lower()
        
        # Detect intent from question
        wants_tests = any(word in question_lower for word in ["тест", "экзамен", "проверка", "оценка", "сдать", "пройти"])
        wants_materials = any(word in question_lower for word in ["материал", "лекция", "учебник", "читать", "изучить", "книга", "конспект"])
        wants_courses = any(word in question_lower for word in ["курс", "предмет", "дисциплина", "обучение", "занятие"])
        wants_videos = any(word in question_lower for word in ["видео", "ролик", "смотреть", "посмотреть"])
        
        # If no specific intent detected, fetch everything
        fetch_all = not (wants_tests or wants_materials or wants_courses or wants_videos)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get courses/subjects (always useful for context)
            if fetch_all or wants_courses:
                try:
                    subjects_response = await client.get(f"{SUBJECT_SERVICE_URL}/subjects")
                    if subjects_response.status_code == 200:
                        subjects = subjects_response.json()
                        if subjects:
                            context_parts.append("\n📚 Доступные курсы:")
                            for subj in subjects[:10]:
                                name = subj.get("name", "")
                                desc = subj.get("description", "")
                                context_parts.append(f"- {name}" + (f": {desc[:100]}" if desc else ""))
                        else:
                            context_parts.append("\n📚 Курсы: пока нет созданных курсов.")
                except Exception as e:
                    logger.warning(f"Failed to fetch subjects: {e}")
            
            # Get materials - either for specific subject or all
            if fetch_all or wants_materials:
                try:
                    params = {"subject_id": request.subject_id} if request.subject_id else {}
                    materials_response = await client.get(f"{MATERIAL_SERVICE_URL}/materials", params=params)
                    if materials_response.status_code == 200:
                        materials = materials_response.json()
                        if materials:
                            context_parts.append("\n📖 Содержание учебных материалов:")
                            # Fetch text content for the first 3 materials to keep context manageable
                            for i, material in enumerate(materials[:3]):
                                mat_id = material.get("id")
                                name = material.get("name", "")
                                note = material.get("note", "")
                                
                                context_parts.append(f"--- Материал: {name} " + (f"({note})" if note else "") + " ---")
                                
                                try:
                                    text_response = await client.get(f"{MATERIAL_SERVICE_URL}/materials/{mat_id}/text")
                                    if text_response.status_code == 200:
                                        text_content = text_response.json().get("text", "")
                                        # Include first 2000 characters of text
                                        context_parts.append(text_content[:2000] + ("..." if len(text_content) > 2000 else ""))
                                    else:
                                        annotation = material.get("annotation", "")
                                        if annotation:
                                            context_parts.append(f"Аннотация: {annotation}")
                                except Exception as te:
                                    logger.warning(f"Failed to fetch text for material {mat_id}: {te}")
                        else:
                            context_parts.append("\n📖 Материалы: пока нет загруженных материалов.")
                except Exception as e:
                    logger.warning(f"Failed to fetch materials: {e}")
            
            # Get videos - either for specific subject or all
            if fetch_all or wants_videos:
                try:
                    params = {"subject_id": request.subject_id} if request.subject_id else {}
                    videos_response = await client.get(f"{VIDEO_SERVICE_URL}/videos", params=params)
                    if videos_response.status_code == 200:
                        videos = videos_response.json()
                        if videos:
                            context_parts.append("\n🎥 Доступные видео:")
                            for video in videos[:10]:
                                title = video.get("title", "")
                                note = video.get("note", "")
                                context_parts.append(f"- {title}" + (f" ({note})" if note else ""))
                        else:
                            context_parts.append("\n🎥 Видео: пока нет добавленных видео.")
                except Exception as e:
                    logger.warning(f"Failed to fetch videos: {e}")
            
            # Get tests - either for specific subject or all
            if fetch_all or wants_tests:
                try:
                    params = {"subject_id": request.subject_id} if request.subject_id else {}
                    tests_response = await client.get(f"{TEST_SERVICE_URL}/tests", params=params)
                    if tests_response.status_code == 200:
                        tests = tests_response.json()
                        if tests:
                            context_parts.append("\n📝 Доступные тесты:")
                            for test in tests[:10]:
                                title = test.get("title", "")
                                due_date = test.get("due_date", "")
                                test_type = test.get("test_type", "")
                                max_attempts = test.get("max_attempts", "")
                                context_parts.append(f"- {title}" + (f" (до {due_date})" if due_date else "") + (f" [{test_type}]" if test_type else ""))
                                if max_attempts:
                                    context_parts.append(f"  Попыток: {max_attempts}")
                        else:
                            context_parts.append("\n📝 Тесты: пока нет созданных тестов.")
                except Exception as e:
                    logger.warning(f"Failed to fetch tests: {e}")
        
        # Format context
        if context_parts:
            context_text = "\n".join(context_parts)
        else:
            context_text = "В системе пока нет данных о курсах, материалах или тестах."
        
        # Create appropriate system prompt based on question type
        if wants_tests:
            system_msg = (
                "Ты - помощник студента по тестам и экзаменам. "
                "Отвечай на вопросы о доступных тестах, сроках сдачи и правилах прохождения. "
                "Используй ТОЛЬКО информацию из контекста. "
                "Отвечай на русском языке."
            )
        elif wants_materials:
            system_msg = (
                "Ты - помощник студента по учебным материалам. "
                "Рекомендуй материалы для изучения на основе контекста. "
                "Используй ТОЛЬКО информацию из контекста. "
                "Отвечай на русском языке."
            )
        elif wants_courses:
            system_msg = (
                "Ты - помощник студента по курсам и обучению. "
                "Расскажи о доступных курсах и их содержании. "
                "Используй ТОЛЬКО информацию из контекста. "
                "Отвечай на русском языке."
            )
        elif any(word in question_lower for word in ["дедлайн", "срок", "когда", "до какого"]):
            system_msg = (
                "Ты - помощник студента. Анализируй дедлайны тестов и заданий. "
                "Отвечай на русском языке, будь конкретным и полезным."
            )
        else:
            system_msg = (
                "Ты - умный помощник студента образовательной платформы. "
                "Отвечай на вопросы, используя информацию из контекста. "
                "Если информации нет в контексте, честно скажи об этом. "
                "Отвечай на русском языке, будь полезным и дружелюбным."
            )
        
        user_msg = f"""
Вопрос студента: {request.question}

Данные из системы:
{context_text}

Ответь на вопрос студента, используя информацию выше. Если данных недостаточно, объясни это понятно.
"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        result = await chat_completion(messages, temperature=0.7, max_tokens=1000)
        
        if not result:
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        return {"message": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat assistant: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


class GenerateTestRequest(BaseModel):
    title: str
    subject_id: str
    description: Optional[str] = ""
    question_count: int
    material_ids: List[str]  # List of material UUIDs
    test_type: str = "multiple_choice"  # "multiple_choice" or "keyword_based"
    additional_conditions: Optional[str] = ""  # Additional prompt conditions


@router.post("/generate-test")
async def generate_test(request: GenerateTestRequest):
    """Generate a test based on materials"""
    try:
        # Fetch materials and extract text
        materials_text = ""
        material_ids_list = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for material_id in request.material_ids:
                try:
                    # Get material text
                    response = await client.get(
                        f"{MATERIAL_SERVICE_URL}/materials/{material_id}/text"
                    )
                    if response.status_code == 200:
                        material_data = response.json()
                        material_text = material_data.get("text", "")
                        if material_text and not material_text.startswith("Error"):
                            materials_text += f"\n\n--- Материал {material_id} ---\n{material_text[:2000]}"  # Limit each material
                            material_ids_list.append(material_id)
                except Exception as e:
                    logger.warning(f"Failed to fetch material {material_id}: {e}")
                    continue
        
        if not materials_text.strip():
            raise HTTPException(status_code=400, detail="No valid material text found")
        
        # Base prompt based on test type
        if request.test_type == "keyword_based":
            system_msg = (
                "Ты - эксперт по созданию образовательных тестов. "
                "Создай тест с вопросами, требующими развернутых ответов, на основе предоставленных материалов. "
                "Для каждого вопроса определи ключевые слова, которые должны присутствовать в правильном ответе студента. "
                "Вопросы должны проверять понимание материала, а не просто запоминание. "
                "Отвечай строго в JSON формате."
            )
            
            user_msg = f"""
Создай тест со следующими параметрами:

Название: {request.title}
Описание: {request.description}
Количество вопросов: {request.question_count}

Материалы для генерации вопросов:
{materials_text}

Требования:
1. Создай {request.question_count} вопросов, требующих развернутых ответов
2. Для каждого вопроса определи 3-5 ключевых слов/фраз, которые должны присутствовать в правильном ответе
3. Каждому ключевому слову присвой количество баллов (всего баллов за вопрос должно быть 10-20)
4. Вопросы должны быть разного уровня сложности
5. Вопросы должны проверять понимание, а не запоминание
6. Используй информацию из предоставленных материалов
"""
            
            if request.additional_conditions:
                user_msg += f"\n\nДополнительные требования:\n{request.additional_conditions}\n"
            
            user_msg += """
Формат ответа (строго JSON):
{
  "questions": [
    {
      "question_id": "q1",
      "title": "Текст вопроса",
      "max_points": 15,
      "keywords": [
        {"word": "ключевое слово 1", "points": 5},
        {"word": "ключевое слово 2", "points": 5},
        {"word": "ключевое слово 3", "points": 5}
      ]
    }
  ]
}
"""
        else:  # multiple_choice
            system_msg = (
                "Ты - эксперт по созданию образовательных тестов. "
                "Создай тест с вариантами ответов на основе предоставленных материалов. "
                "Каждый вопрос должен иметь 4 варианта ответа, один из которых правильный. "
                "Вопросы должны проверять понимание материала, а не просто запоминание. "
                "Отвечай строго в JSON формате."
            )
            
            user_msg = f"""
Создай тест со следующими параметрами:

Название: {request.title}
Описание: {request.description}
Количество вопросов: {request.question_count}

Материалы для генерации вопросов:
{materials_text}

Требования:
1. Создай {request.question_count} вопросов с вариантами ответов
2. Каждый вопрос должен иметь 4 варианта ответа (A, B, C, D)
3. Один вариант должен быть правильным
4. Вопросы должны быть разного уровня сложности
5. Вопросы должны проверять понимание, а не запоминание
6. Используй информацию из предоставленных материалов
"""
            
            if request.additional_conditions:
                user_msg += f"\n\nДополнительные требования:\n{request.additional_conditions}\n"
            
            user_msg += """
Формат ответа (строго JSON):
{
  "questions": [
    {
      "question_id": "q1",
      "title": "Текст вопроса",
      "options": ["Вариант A", "Вариант B", "Вариант C", "Вариант D"],
      "correct_answer": "Вариант A",
      "max_points": 10
    }
  ]
}
"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        result = await chat_completion(messages, temperature=0.7, max_tokens=4000)
        
        if not result:
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                test_data = json.loads(json_str)
                # Validate structure
                if "questions" not in test_data:
                    raise HTTPException(status_code=500, detail="Invalid test structure from AI")
                
                # Return test data ready for Test Service
                return {
                    "test_type": request.test_type,
                    "questions": test_data["questions"],
                    "material_ids": material_ids_list  # Store material IDs for feedback
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response: {e}")
                raise HTTPException(status_code=500, detail="Failed to parse AI response")
        else:
            raise HTTPException(status_code=500, detail="No JSON found in AI response")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating test: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating test: {str(e)}")


class TestFeedbackRequest(BaseModel):
    test_id: str
    test_type: str
    question_id: str
    question_title: str
    student_answer: str
    correct_answer: Optional[str] = None
    material_ids: Optional[List[str]] = None  # For multiple_choice AI tests
    keywords: Optional[List[Dict]] = None  # For keyword_based tests
    max_points: int


class TestFeedbackResponse(BaseModel):
    feedback: Dict  # Contains feedback data based on test type


@router.post("/test-feedback")
async def get_test_feedback(request: TestFeedbackRequest):
    """Get AI feedback for test submission"""
    try:
        if request.test_type == "multiple_choice":
            # For AI-generated multiple choice tests: show materials and correct answer from materials
            if not request.material_ids:
                return {
                    "feedback": {
                        "type": "multiple_choice",
                        "materials_info": None,
                        "material_answers": None,
                        "message": "Этот тест не был сгенерирован AI или материалы недоступны"
                    }
                }
            
            # Fetch materials text
            materials_info = []
            material_answers = []
            async with httpx.AsyncClient(timeout=30.0) as client:
                for material_id in request.material_ids:
                    try:
                        # Get material info
                        material_response = await client.get(
                            f"{MATERIAL_SERVICE_URL}/materials/{material_id}"
                        )
                        if material_response.status_code == 200:
                            material_data = material_response.json()
                            materials_info.append({
                                "id": material_id,
                                "name": material_data.get("name", ""),
                                "original_name": material_data.get("original_name", "")
                            })
                        
                        # Get material text for finding answer
                        text_response = await client.get(
                            f"{MATERIAL_SERVICE_URL}/materials/{material_id}/text"
                        )
                        if text_response.status_code == 200:
                            text_data = text_response.json()
                            material_text = text_data.get("text", "")
                            if material_text and not material_text.startswith("Error"):
                                # Use AI to extract relevant answer from material
                                system_msg = (
                                    "Ты - помощник для образовательной платформы. "
                                    "Найди в предоставленном материале информацию, которая отвечает на вопрос. "
                                    "Верни краткий, но информативный ответ (2-3 предложения). "
                                    "Если информации нет, верни 'Информация не найдена в данном материале'."
                                )
                                
                                user_msg = f"""
Вопрос: {request.question_title}

Правильный ответ: {request.correct_answer}

Материал:
{material_text[:3000]}

Найди в материале информацию, которая объясняет правильный ответ на этот вопрос.
"""
                                
                                messages = [
                                    {"role": "system", "content": system_msg},
                                    {"role": "user", "content": user_msg}
                                ]
                                
                                answer_from_material = await chat_completion(messages, temperature=0.3, max_tokens=300)
                                if answer_from_material:
                                    material_answers.append({
                                        "material_id": material_id,
                                        "material_name": material_data.get("name", ""),
                                        "answer": answer_from_material
                                    })
                    except Exception as e:
                        logger.warning(f"Failed to fetch material {material_id} for feedback: {e}")
                        continue
            
            return {
                "feedback": {
                    "type": "multiple_choice",
                    "materials_info": materials_info,
                    "material_answers": material_answers,
                    "is_correct": request.student_answer == request.correct_answer
                }
            }
        
        elif request.test_type == "keyword_based":
            # For keyword-based tests: evaluate answer and provide recommended score
            if not request.keywords:
                return {
                    "feedback": {
                        "type": "keyword_based",
                        "message": "Ключевые слова не найдены"
                    }
                }
            
            keywords_text = ", ".join([kw.get("word", "") for kw in request.keywords])
            max_keyword_points = sum([kw.get("points", 0) for kw in request.keywords])
            
            system_msg = (
                "Ты - строгий, но справедливый преподаватель. "
                "Оцени ответ студента на вопрос с ключевыми словами. "
                "Проверь, используются ли ключевые слова в ответе, и имеют ли они смысл в контексте. "
                "Если ключевые слова есть, но не имеют смысла или используются неправильно, ставь 0 баллов. "
                "Если ключевые слова есть и используются правильно, ставь баллы пропорционально количеству найденных ключевых слов. "
                "Верни строго JSON формат."
            )
            
            user_msg = f"""
Вопрос: {request.question_title}

Ключевые слова и их баллы:
{chr(10).join([f"- {kw.get('word', '')}: {kw.get('points', 0)} баллов" for kw in request.keywords])}

Максимум баллов за вопрос: {request.max_points}
Максимум баллов за ключевые слова: {max_keyword_points}

Ответ студента:
{request.student_answer}

Оцени ответ студента:
1. Проверь, какие ключевые слова присутствуют в ответе
2. Проверь, имеют ли эти слова смысл в контексте ответа
3. Если слова есть, но не имеют смысла - ставь 0 баллов
4. Если слова есть и используются правильно - ставь баллы пропорционально
5. Учти общее качество ответа и понимание темы

Верни JSON в формате:
{{
  "recommended_score": <число от 0 до max_points>,
  "found_keywords": ["слово1", "слово2"],
  "missing_keywords": ["слово3", "слово4"],
  "evaluation": "Подробная оценка ответа студента",
  "feedback": "Конструктивная обратная связь для студента"
}}
"""
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            
            result = await chat_completion(messages, temperature=0.3, max_tokens=800)
            
            if not result:
                raise HTTPException(status_code=503, detail="AI service unavailable")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    feedback_data = json.loads(json_str)
                    recommended_score = min(request.max_points, max(0, int(feedback_data.get("recommended_score", 0))))
                    
                    return {
                        "feedback": {
                            "type": "keyword_based",
                            "recommended_score": recommended_score,
                            "found_keywords": feedback_data.get("found_keywords", []),
                            "missing_keywords": feedback_data.get("missing_keywords", []),
                            "evaluation": feedback_data.get("evaluation", ""),
                            "feedback": feedback_data.get("feedback", "")
                        }
                    }
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse AI feedback response: {e}")
                    # Return basic feedback if parsing fails
                    return {
                        "feedback": {
                            "type": "keyword_based",
                            "recommended_score": 0,
                            "found_keywords": [],
                            "missing_keywords": [kw.get("word", "") for kw in request.keywords],
                            "evaluation": "Не удалось обработать ответ AI",
                            "feedback": "Проверьте, что в вашем ответе присутствуют все ключевые слова из вопроса"
                        }
                    }
            else:
                raise HTTPException(status_code=500, detail="No JSON found in AI feedback response")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported test type: {request.test_type}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting test feedback: {str(e)}")


class GenerateCourseRequest(BaseModel):
    topic: str
    target_audience: str = "Beginners"
    additional_info: Optional[str] = None


@router.post("/generate-course")
async def generate_course(request: GenerateCourseRequest):
    """Generate a full course structure using AI"""
    try:
        # Prompt for GigaChat
        system_msg = (
            "Ты - методист и создатель образовательных курсов. "
            "Твоя задача - создать структуру курса по заданной теме. "
            "Структура должна включать модули и уроки. "
            "Для каждого урока создай краткое текстовое содержание (3-4 абзаца). "
            "Отвечай строго в JSON формате."
        )
        
        user_msg = f"""
Создай структуру курса по теме: "{request.topic}"
Целевая аудитория: {request.target_audience}
{f"Дополнительная информация: {request.additional_info}" if request.additional_info else ""}

ВАЖНО: Ответ должен быть КОРОТКИМ и уместиться в лимит токенов!

Требования:
1. Курс должен состоять из РОВНО 2 модулей.
2. В каждом модуле должно быть РОВНО 2 урока.
3. Для каждого урока напиши "content_text" - обучающий материал объемом 500-800 знаков (не более!).
4. Используй простой и понятный язык.

Формат ответа (строго JSON, БЕЗ дополнительного текста):
{{
  "title": "Название курса",
  "description": "Краткое описание курса",
  "modules": [
    {{
      "title": "Название модуля 1",
      "description": "Описание модуля",
      "lessons": [
        {{
          "title": "Название урока 1.1",
          "content_text": "Текст урока..."
        }},
        {{
          "title": "Название урока 1.2",
          "content_text": "Текст урока..."
        }}
      ]
    }},
    {{
      "title": "Название модуля 2",
      "description": "Описание модуля",
      "lessons": [
        {{
          "title": "Название урока 2.1",
          "content_text": "Текст урока..."
        }},
        {{
          "title": "Название урока 2.2",
          "content_text": "Текст урока..."
        }}
      ]
    }}
  ]
}}
"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # Reduced max_tokens to prevent truncation
        result = await chat_completion(messages, temperature=0.7, max_tokens=4000)
        
        if not result:
            raise HTTPException(status_code=503, detail="AI service unavailable")
        
        # Log the raw response for debugging
        logger.info(f"Raw AI response (first 500 chars): {result[:500]}")
        
        # Extract JSON - try to find it within code blocks first
        json_str = None
        
        # Try to extract from ```json code block
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Fallback: find any JSON object
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        
        if not json_str:
            logger.error(f"No JSON found in AI response: {result}")
            raise HTTPException(status_code=500, detail="No JSON found in AI response")
            
        try:
            course_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.error(f"JSON string that failed: {json_str[:1000]}")
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
            
        # Create Course via Subject Service
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1. Create Subject
            subject_payload = {
                "name": course_data.get("title", request.topic),
                "description": course_data.get("description", f"Generated course on {request.topic}")
            }
            resp = await client.post(f"{SUBJECT_SERVICE_URL}/subjects", json=subject_payload)
            if resp.status_code not in [200, 201]:
                raise HTTPException(status_code=500, detail=f"Failed to create subject: {resp.text}")
            subject = resp.json()
            subject_id = subject["id"]
            
            # 2. Iterate Modules
            modules = course_data.get("modules", [])
            for mod_idx, mod in enumerate(modules):
                mod_payload = {
                    "title": mod.get("title", f"Module {mod_idx+1}"),
                    "description": mod.get("description", ""),
                    "order_index": mod_idx
                }
                resp = await client.post(f"{SUBJECT_SERVICE_URL}/subjects/{subject_id}/modules", json=mod_payload)
                if resp.status_code not in [200, 201]:
                    logger.error(f"Failed to create module: {resp.text}")
                    continue
                module = resp.json()
                module_id = module["id"]
                
                # 3. Iterate Lessons
                lessons = mod.get("lessons", [])
                for lesson_idx, lesson in enumerate(lessons):
                    lesson_payload = {
                        "title": lesson.get("title", f"Lesson {lesson_idx+1}"),
                        "lesson_type": "lecture",
                        "order_index": lesson_idx
                    }
                    resp = await client.post(f"{SUBJECT_SERVICE_URL}/modules/{module_id}/lessons", json=lesson_payload)
                    if resp.status_code not in [200, 201]:
                        logger.error(f"Failed to create lesson: {resp.text}")
                        continue
                    new_lesson = resp.json()
                    lesson_id = new_lesson["id"]
                    
                    # 4. Create Content
                    content_text = lesson.get("content_text", "")
                    if content_text:
                        content_payload = {
                            "text_content": content_text,
                            "lesson_id": lesson_id
                        }
                        # Check endpoint: PUT /lessons/{id}/content or POST?
                        # Based on checked code: POST /lessons/{lessonId}/content
                        resp = await client.post(f"{SUBJECT_SERVICE_URL}/lessons/{lesson_id}/content", json=content_payload)
                        if resp.status_code not in [200, 201]:
                            logger.error(f"Failed to create content: {resp.text}")

        return {"message": "Course generated successfully", "subject_id": subject_id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating course: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating course: {str(e)}")
