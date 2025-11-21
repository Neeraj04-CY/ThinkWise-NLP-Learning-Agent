import os
import logging
import re
import uuid
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, List, Dict, Any, Union

from flask import Flask, request, jsonify, session, render_template, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import openai
from pydantic import BaseModel, ValidationError, validator
import json
from dotenv import load_dotenv
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
import redis
from celery import Celery
import requests

# Load environment variables
load_dotenv()

# Enhanced Configuration
class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    JWT_SECRET = os.environ.get('JWT_SECRET', SECRET_KEY)
    
    # AI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
    AI_MODEL = os.environ.get('AI_MODEL', 'gpt-3.5-turbo')
    FALLBACK_MODEL = os.environ.get('FALLBACK_MODEL', 'gpt-3.5-turbo')
    MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 2000))
    TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.7))
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///thinkwise.db')
    
    # Redis for caching and rate limiting
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Rate Limiting
    RATE_LIMIT = os.environ.get('RATE_LIMIT', '100 per hour')
    AI_RATE_LIMIT = os.environ.get('AI_RATE_LIMIT', '30 per hour')
    
    # CORS
    ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:5000').split(',')
    
    # Feature Flags
    ENABLE_ANALYTICS = os.environ.get('ENABLE_ANALYTICS', 'true').lower() == 'true'
    ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'true').lower() == 'true'
    ENABLE_QUEUE = os.environ.get('ENABLE_QUEUE', 'false').lower() == 'true'

# Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
# Ensure Flask-SQLAlchemy sees the expected config key
if 'SQLALCHEMY_DATABASE_URI' not in app.config and hasattr(Config, 'DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.DATABASE_URL

# Database Setup
Base = declarative_base()
db = SQLAlchemy(app)

# Redis for caching
redis_client = None
_limiter_storage_uri = "memory://"
try:
    if Config.REDIS_URL:
        candidate = redis.from_url(Config.REDIS_URL)
        # quick health check
        try:
            candidate.ping()
            redis_client = candidate
            _limiter_storage_uri = Config.REDIS_URL
        except redis.RedisError:
            # couldn't reach redis; continue with in-memory fallback
            redis_client = None
            _limiter_storage_uri = "memory://"
    else:
        _limiter_storage_uri = "memory://"
except Exception:
    # Any unexpected error: disable redis usage and use in-memory limiter
    redis_client = None
    _limiter_storage_uri = "memory://"

# Celery for async tasks (fall back to in-memory broker/backend when Redis unavailable)
_celery_broker = Config.REDIS_URL if redis_client else "memory://"
_celery_backend = Config.REDIS_URL if redis_client else "memory://"
celery = Celery(
    app.name,
    broker=_celery_broker,
    backend=_celery_backend
)
celery.conf.update(app.config)

# Security & Rate Limiting
CORS(app, origins=Config.ALLOWED_ORIGINS, supports_credentials=True)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT],
    storage_uri=_limiter_storage_uri
)

# Initialize AI Clients
if Config.OPENAI_API_KEY:
    openai.api_key = Config.OPENAI_API_KEY

# Database Models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255))
    learning_style = Column(String(50), default='visual')
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)

class StudySession(Base):
    __tablename__ = 'study_sessions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False)
    topic = Column(String(255), nullable=False)
    duration_minutes = Column(Integer, default=0)
    activity_type = Column(String(50))  # concept_breakdown, quiz, notes
    complexity = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    # 'metadata' is a reserved attribute name in SQLAlchemy's Declarative API,
    # so map the DB column named 'metadata' to the attribute `metadata_json`.
    metadata_json = Column('metadata', JSON)  # Store additional data like AI response, scores, etc.

class UserProgress(Base):
    __tablename__ = 'user_progress'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False)
    subject = Column(String(255), nullable=False)
    topic = Column(String(255), nullable=False)
    proficiency_score = Column(Float, default=0.0)  # 0-100
    times_studied = Column(Integer, default=0)
    last_studied = Column(DateTime, default=datetime.utcnow)
    weak_areas = Column(JSON)  # List of weak areas in this topic
    strengths = Column(JSON)   # List of strengths in this topic

class CachedResponse(Base):
    __tablename__ = 'cached_responses'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    cache_key = Column(String(512), unique=True, nullable=False)
    response_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    hit_count = Column(Integer, default=0)

# Enhanced Request Models with Validation
class ConceptBreakdownRequest(BaseModel):
    topic: str
    complexity: str = "beginner"
    learning_style: Optional[str] = "visual"
    session_id: Optional[str] = None
    
    @validator('topic')
    def topic_must_be_valid(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Topic must be at least 2 characters long')
        if len(v.strip()) > 500:
            raise ValueError('Topic too long')
        return v.strip()
    
    @validator('complexity')
    def complexity_must_be_valid(cls, v):
        if v not in ['beginner', 'intermediate', 'advanced']:
            raise ValueError('Complexity must be beginner, intermediate, or advanced')
        return v

class QuestionSolverRequest(BaseModel):
    question: str
    subject: str
    exam_level: Optional[str] = "high_school"
    show_steps: Optional[bool] = True
    
    @validator('question')
    def question_must_be_valid(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Question must be at least 5 characters long')
        return v.strip()

class StudyMaterialsRequest(BaseModel):
    content: str
    content_type: str
    target_length: Optional[str] = "concise"
    include_quiz: Optional[bool] = False
    
    @validator('content_type')
    def content_type_must_be_valid(cls, v):
        valid_types = ['pdf_text', 'youtube_transcript', 'textbook', 'web_article']
        if v not in valid_types:
            raise ValueError(f'Content type must be one of: {", ".join(valid_types)}')
        return v

class QuizGenerationRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    question_count: int = 5
    question_types: Optional[List[str]] = None
    
    @validator('question_count')
    def question_count_must_be_valid(cls, v):
        if not 1 <= v <= 20:
            raise ValueError('Question count must be between 1 and 20')
        return v


# Pydantic models for validating multi-depth module output from LLM
class MultiDepthExample(BaseModel):
    input: str
    steps: List[str]
    answer: str


class MediumSection(BaseModel):
    explanation: str
    examples: List[MultiDepthExample]
    citations: Optional[List[str]] = None


class DeepSection(BaseModel):
    outline: str
    sections: List[str]
    citations: Optional[List[str]] = None


class MultiDepthModule(BaseModel):
    topic: str
    short: str
    medium: MediumSection
    deep: DeepSection
    generated_at: Optional[str]
    ai_model: Optional[str]

# Authentication Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split(' ')[1]
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, Config.JWT_SECRET, algorithms=['HS256'])
            current_user = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# Utility Functions
def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate a unique cache key from parameters"""
    sorted_params = sorted(kwargs.items())
    param_str = ':'.join(f"{k}={v}" for k, v in sorted_params)
    return f"{prefix}:{hash(param_str)}"

def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response from Redis or database"""
    if not Config.ENABLE_CACHING:
        return None
    
    # Try Redis first
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except redis.RedisError:
            pass
    
    # Fallback to database
    try:
        cached_response = CachedResponse.query.filter_by(cache_key=cache_key).first()
        if cached_response and cached_response.expires_at > datetime.utcnow():
            cached_response.hit_count += 1
            db.session.commit()
            return cached_response.response_data
    except Exception:
        pass
    
    return None

def set_cached_response(cache_key: str, data: Dict, expire_hours: int = 24):
    """Cache response in Redis and database"""
    if not Config.ENABLE_CACHING:
        return
    
    expires_at = datetime.utcnow() + timedelta(hours=expire_hours)
    
    # Cache in Redis
    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                expire_hours * 3600,
                json.dumps(data)
            )
        except redis.RedisError:
            pass
    
    # Cache in database as fallback
    try:
        cached_response = CachedResponse.query.filter_by(cache_key=cache_key).first()
        if cached_response:
            cached_response.response_data = data
            cached_response.expires_at = expires_at
            cached_response.hit_count += 1
        else:
            cached_response = CachedResponse(
                cache_key=cache_key,
                response_data=data,
                expires_at=expires_at
            )
            db.session.add(cached_response)
        db.session.commit()
    except Exception as e:
        logging.warning(f"Failed to cache in database: {e}")

def track_user_activity(user_id: str, activity_type: str, topic: str, complexity: str = None, metadata: Dict = None):
    """Track user study activity"""
    try:
        session = StudySession(
            user_id=user_id,
            topic=topic,
            activity_type=activity_type,
            complexity=complexity,
            metadata_json=metadata or {}
        )
        db.session.add(session)
        
        # Update user progress
        progress = UserProgress.query.filter_by(user_id=user_id, topic=topic).first()
        if progress:
            progress.times_studied += 1
            progress.last_studied = datetime.utcnow()
        else:
            progress = UserProgress(
                user_id=user_id,
                subject=topic.split()[0].lower() if topic else 'general',
                topic=topic,
                times_studied=1
            )
            db.session.add(progress)
        
        db.session.commit()
    except Exception as e:
        logging.error(f"Failed to track user activity: {e}")
        db.session.rollback()

def extract_json_from_text(text: str) -> Optional[str]:
    """Enhanced JSON extraction with multiple fallback methods"""
    if not text:
        return None
    
    # Method 1: Try to find JSON in code blocks
    json_patterns = [
        r"```json\s*(\{.*\})\s*```",
        r"```\s*(\{.*\})\s*```",
        r"```.*?\n(\{.*\})\n```",
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    
    # Method 2: Find first balanced JSON object
    start = text.find('{')
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
    
    return None

# Enhanced AI Service Layer
class AIService:
    @staticmethod
    def generate_concept_breakdown(topic: str, complexity: str, learning_style: str) -> Dict[str, Any]:
        """Generate comprehensive concept breakdown using AI with enhanced prompts"""
        
        # Check cache first
        cache_key = generate_cache_key(
            "concept_breakdown",
            topic=topic,
            complexity=complexity,
            learning_style=learning_style
        )
        
        cached = get_cached_response(cache_key)
        if cached:
            logging.info("Returning cached concept breakdown")
            return cached
        
        # If no API key, return enhanced mock data
        if not Config.OPENAI_API_KEY:
            logging.info("OPENAI_API_KEY not set — returning enhanced mock breakdown")
            mock_data = AIService._generate_mock_breakdown(topic, complexity)
            set_cached_response(cache_key, mock_data, 1)  # Cache mock for 1 hour
            return mock_data
        
        # Enhanced prompt for better structured responses
        prompt = AIService._build_concept_prompt(topic, complexity, learning_style)
        
        try:
            response = openai.ChatCompletion.create(
                model=Config.AI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert educator who creates clear, engaging, and structured learning content. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                response_format={ "type": "json_object" }
            )
            
            ai_content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(ai_content)
                
                # Validate required fields
                required_fields = ['topic_meaning', 'why_it_matters', 'key_concepts', 'summary']
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                
                # Enhance with additional metadata
                result['ai_model'] = Config.AI_MODEL
                result['generated_at'] = datetime.utcnow().isoformat()
                
                # Cache successful response
                set_cached_response(cache_key, result, 24)
                
                return result
                
            except json.JSONDecodeError:
                # Fallback: try to extract JSON
                extracted_json = extract_json_from_text(ai_content)
                if extracted_json:
                    result = json.loads(extracted_json)
                    set_cached_response(cache_key, result, 24)
                    return result
                else:
                    raise ValueError("AI response is not valid JSON")
                    
        except Exception as e:
            logging.error(f"AI API Error: {e}")
            
            # Fallback to mock data in case of error
            if Config.OPENAI_API_KEY:
                logging.info("Falling back to mock data due to API error")
                mock_data = AIService._generate_mock_breakdown(topic, complexity)
                return mock_data
            else:
                raise Exception(f"AI generation failed: {str(e)}")
    
    @staticmethod
    def _build_concept_prompt(topic: str, complexity: str, learning_style: str) -> str:
        """Build enhanced prompt for concept breakdown"""
        
        complexity_descriptions = {
            "beginner": "Explain like I'm completely new to this, with simple analogies and basic examples",
            "intermediate": "Assume I have basic knowledge but want to deepen my understanding",
            "advanced": "Provide detailed, technical explanations with advanced applications"
        }
        
        learning_style_guides = {
            "visual": "Include visual metaphors and spatial relationships",
            "auditory": "Focus on rhythmic patterns and verbal explanations", 
            "kinesthetic": "Use hands-on analogies and practical applications",
            "reading": "Provide clear, well-structured written explanations"
        }
        
        return f"""
        Create a comprehensive learning breakdown for: "{topic}"
        
        Audience: {complexity_descriptions.get(complexity, 'beginner')}
        Learning Style: {learning_style_guides.get(learning_style, 'visual')}
        
        Return ONLY valid JSON with this exact structure:
        {{
            "topic_meaning": "Clear definition (2-3 sentences)",
            "why_it_matters": "Practical importance and real-world applications",
            "real_life_analogy": "Creative analogy that makes it relatable",
            "key_concepts": ["List", "of", "3-5", "fundamental", "ideas"],
            "step_by_step_explanation": ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
            "common_misconceptions": ["List", "of", "common", "misunderstandings"],
            "summary": ["Bullet 1", "Bullet 2", "Bullet 3", "Bullet 4", "Bullet 5"],
            "suggested_next_topics": ["Related", "topics", "to study", "next"],
            "learning_tips": ["Practical tips", "for mastering", "this topic"]
        }}
        
        Make the response engaging, practical, and tailored to {complexity} level.
        """

    @staticmethod
    def _generate_mock_breakdown(topic: str, complexity: str) -> Dict[str, Any]:
        """Generate enhanced mock data for development"""
        return {
            "topic_meaning": f"{topic} is a fundamental concept that forms the basis for more advanced learning. At its core, it deals with essential principles that are widely applicable across various domains.",
            "why_it_matters": f"Understanding {topic} is crucial because it appears in real-world applications from technology to daily life. Mastering this concept will help you solve complex problems and build a strong foundation for advanced topics.",
            "real_life_analogy": f"Think of {topic} like learning to cook - you start with basic recipes, understand ingredients, and gradually create complex dishes. Each step builds on previous knowledge, just like mastering {topic}.",
            "key_concepts": [
                f"Fundamental Principle of {topic}",
                "Core Components and Structure", 
                "Practical Applications",
                "Common Variations",
                "Advanced Extensions"
            ],
            "step_by_step_explanation": [
                f"Start with the basic definition of {topic}",
                "Understand the core components and how they interact",
                "Learn through practical examples and applications", 
                "Practice with progressively challenging problems",
                "Connect to related concepts for deeper understanding"
            ],
            "common_misconceptions": [
                "Thinking it's more complicated than it actually is",
                "Confusing similar but distinct concepts",
                "Overlooking practical applications"
            ],
            "summary": [
                f"{topic} is built on fundamental principles",
                "It has wide-ranging practical applications",
                "Mastery comes through practice and application",
                "Common misconceptions can hinder learning",
                "Connections to other topics enhance understanding"
            ],
            "suggested_next_topics": [
                f"Advanced {topic} Applications",
                f"{topic} in Real-World Contexts",
                "Related Complementary Concepts",
                "Historical Development of {topic}"
            ],
            "learning_tips": [
                "Practice with real examples daily",
                "Create mind maps to visualize connections",
                "Teach the concept to someone else",
                "Relate to your existing knowledge"
            ],
            "ai_model": "mock",
            "generated_at": datetime.utcnow().isoformat()
        }

    @staticmethod
    def solve_question(question: str, subject: str, exam_level: str, show_steps: bool = True) -> Dict[str, Any]:
        """Solve educational questions with step-by-step explanations"""
        # Implementation similar to generate_concept_breakdown
        # ... (would include similar caching, error handling, etc.)
        return {
            "question": question,
            "subject": subject,
            "solution": "Detailed solution would appear here",
            "steps": ["Step 1", "Step 2", "Step 3"] if show_steps else [],
            "final_answer": "The final answer",
            "key_concepts": ["Relevant concept 1", "Relevant concept 2"],
            "common_mistakes": ["Common error 1", "Common error 2"]
        }

    @staticmethod
    def generate_quiz(topic: str, difficulty: str, question_count: int, question_types: List[str] = None) -> Dict[str, Any]:
        """Generate educational quiz with various question types"""
        # Implementation for quiz generation
        return {
            "topic": topic,
            "difficulty": difficulty,
            "questions": [
                {
                    "id": 1,
                    "type": "multiple_choice",
                    "question": f"What is the fundamental principle of {topic}?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": 0,
                    "explanation": "Detailed explanation of why this is correct"
                }
            ],
            "time_limit": 600,  # 10 minutes
            "passing_score": 70
        }

    @staticmethod
    def _build_multi_depth_prompt(topic: str) -> str:
        """Construct the structured prompt for the multi-depth learning module."""

        return f"""
        You are a world-class instructional designer. Build a three-depth learning module for "{topic}".

        Return ONLY valid JSON (no prose) with this exact schema:
        {{
          "topic": "{topic}",
          "short": "2-3 sentence intuition focused on why it matters now",
          "medium": {{
            "explanation": "4-6 sentence accessible explanation that still feels rigorous",
            "examples": [
              {{
                "input": "Concise scenario or problem statement",
                "steps": ["Step 1 reasoning", "Step 2 reasoning", "Step 3 reasoning"],
                "answer": "Short final answer or takeaway"
              }},
              {{
                "input": "Second scenario with a different lens",
                "steps": ["Step 1", "Step 2", "Step 3"],
                "answer": "What we learn here"
              }}
            ],
            "citations": ["Author, Title, Year", "https://credible.source"]
          }},
          "deep": {{
            "outline": "2 sentence summary of the deep dive",
            "sections": [
              "Section 1 - conceptual grounding",
              "Section 2 - worked derivation/code",
              "Section 3 - applications or edge cases",
              "Section 4 - reflection/project prompt"
            ],
            "citations": ["Primary textbook or paper", "Second reputable source"]
          }}
        }}

        Requirements:
        - The response MUST be valid JSON and include every field exactly as named.
        - Provide at least two medium examples; each needs 3-5 short imperative steps showing reasoning.
        - Citations can be scholarly works, textbooks, or reputable sites (real or placeholder but realistic).
        - Keep language motivating but precise, aiming for serious self-learners.
        """

    @staticmethod
    def _build_structured_module_stub(topic: str, note: str, ai_model: str) -> Dict[str, Any]:
        """Create a structured fallback module that satisfies the Pydantic schema."""

        timestamp = datetime.utcnow().isoformat()
        module = {
            "topic": topic,
            "short": f"{topic}: {note} overview that captures the core intuition in under a minute.",
            "medium": {
                "explanation": (
                    f"This {note.lower()} explanation introduces {topic} step-by-step, highlighting when and why to use it, "
                    "then reinforcing the idea with approachable scenarios."
                ),
                "examples": [
                    {
                        "input": f"How would you apply {topic} to a quick everyday decision?",
                        "steps": [
                            f"Identify the pieces of the situation that relate to {topic}.",
                            "Map those pieces to the rule or equation behind the concept.",
                            "Summarize the outcome and what it teaches you."
                        ],
                        "answer": f"Using {topic} clarifies the decision and shows the value of structured thinking."
                    },
                    {
                        "input": f"Show {topic} in a lightweight technical example.",
                        "steps": [
                            "State the givens or inputs.",
                            "Work through the transformation or calculation explained by the concept.",
                            "Verify the result and connect it back to the intuition."
                        ],
                        "answer": f"The process demonstrates the repeatable mechanics behind {topic}."
                    }
                ],
                "citations": [
                    f"{topic} Essentials — Open Learning Notes",
                    "Brown, S. (2021). Modern Learning Patterns"
                ]
            },
            "deep": {
                "outline": (
                    f"A structured deep dive into {topic} that blends theory, derivations, and project-ready prompts so learners "
                    "can immediately apply the concept even without AI output."
                ),
                "sections": [
                    "Foundations and historical context",
                    "Derivations or core mechanics",
                    "Implementation walkthrough or code",
                    "Edge cases, misconceptions, and next steps"
                ],
                "citations": [
                    f"{topic} Handbook (2020)",
                    "MIT OpenCourseWare"
                ]
            },
            "ai_model": ai_model,
            "generated_at": timestamp
        }

        return MultiDepthModule.parse_obj(module).dict()

    @staticmethod
    def generate_multi_depth_module(topic: str) -> Dict[str, Any]:
        """Generate a three-level (short/medium/deep) structured module for a topic.

        Returns a dict with keys: short, medium, deep. Uses mock data when OPENAI_API_KEY not set.
        """

        if not topic:
            raise ValueError("Topic is required")

        # If no API key, return a deterministic mock module and validate it
        if not Config.OPENAI_API_KEY:
            return AIService._build_structured_module_stub(
                topic=topic,
                note="Mock reference module",
                ai_model="mock"
            )

        # Build prompt and call AI
        prompt = AIService._build_multi_depth_prompt(topic)
        try:
            response = openai.ChatCompletion.create(
                model=Config.AI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert educator. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
        except Exception as e:
            logging.warning(f"AI call failed: {e}")
            return AIService._build_structured_module_stub(
                topic=topic,
                note="Fallback after AI call failure",
                ai_model="error"
            )

        ai_content = response.choices[0].message.content

        # Parse AI output as JSON, with a substring fallback
        try:
            result = json.loads(ai_content)
        except Exception:
            extracted = extract_json_from_text(ai_content)
            if extracted:
                try:
                    result = json.loads(extracted)
                except Exception:
                    logging.warning("Extracted JSON could not be parsed")
                    return AIService._build_structured_module_stub(
                        topic=topic,
                        note="Fallback after malformed JSON extraction",
                        ai_model=Config.AI_MODEL
                    )
            else:
                logging.warning("AI output could not be parsed as JSON and no extractable JSON found")
                return AIService._build_structured_module_stub(
                    topic=topic,
                    note="Fallback when no JSON detected",
                    ai_model=Config.AI_MODEL
                )

        # Validate structure using Pydantic model
        result["ai_model"] = Config.AI_MODEL
        result["generated_at"] = datetime.utcnow().isoformat()
        try:
            validated = MultiDepthModule.parse_obj(result)
            return validated.dict()
        except ValidationError as ve:
            logging.warning(f"Validation failed for AI module output: {ve}")
            return {
                "topic": topic,
                "validation_error": str(ve),
                "original_output": result,
                "ai_model": Config.AI_MODEL,
                "generated_at": datetime.utcnow().isoformat()
            }
        

# Celery Tasks for Async Processing
@celery.task(bind=True)
def generate_concept_breakdown_async(self, topic: str, complexity: str, learning_style: str) -> Dict[str, Any]:
    """Async task for concept breakdown generation"""
    try:
        result = AIService.generate_concept_breakdown(topic, complexity, learning_style)
        return {'status': 'SUCCESS', 'result': result}
    except Exception as e:
        return {'status': 'FAILED', 'error': str(e)}

# Enhanced Routes
@app.route('/')
def home():
    """Enhanced home page with analytics"""
    return render_template('index.html')

@app.route('/study/<subject>')
def study(subject):
    """Enhanced study materials page"""
    progress = 35  # Would be calculated from user data
    return render_template('study_materials.html', 
                         subject=subject, 
                         progress=progress,
                         features_enabled=Config.ENABLE_ANALYTICS)

# Authentication Routes
@app.route('/api/v1/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Check if user exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'User already exists'}), 409
        
        # Create new user
        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            name=name
        )
        db.session.add(user)
        db.session.commit()
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, Config.JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user.id,
                'email': user.email,
                'name': user.name
            }
        }), 201
        
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/v1/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, Config.JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user.id,
                'email': user.email,
                'name': user.name
            }
        })
        
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Enhanced API Endpoints
@app.route('/api/v1/breakdown', methods=['POST'])
@limiter.limit(Config.AI_RATE_LIMIT)
def api_concept_breakdown():
    """Enhanced concept breakdown endpoint with async support"""
    try:
        body = request.get_json(force=True)
        data = ConceptBreakdownRequest(**body)
        
        # Get user ID from token if available
        user_id = None
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split(' ')[1]
                payload = jwt.decode(token, Config.JWT_SECRET, algorithms=['HS256'])
                user_id = payload['user_id']
            except (IndexError, jwt.InvalidTokenError):
                pass  # Continue without user context
        
        # Async processing for complex requests
        if Config.ENABLE_QUEUE and data.complexity == 'advanced':
            task = generate_concept_breakdown_async.delay(
                data.topic, data.complexity, data.learning_style
            )
            return jsonify({
                'success': True,
                'task_id': task.id,
                'status': 'processing',
                'message': 'Breakdown is being generated asynchronously'
            }), 202
        
        # Synchronous processing
        breakdown = AIService.generate_concept_breakdown(
            topic=data.topic,
            complexity=data.complexity,
            learning_style=data.learning_style
        )
        
        # Track activity if user is authenticated
        if user_id:
            track_user_activity(
                user_id=user_id,
                activity_type='concept_breakdown',
                topic=data.topic,
                complexity=data.complexity,
                metadata={'learning_style': data.learning_style}
            )
        
        return jsonify({
            "success": True,
            "data": breakdown,
            "metadata": {
                "topic": data.topic,
                "generated_at": datetime.utcnow().isoformat(),
                "complexity_level": data.complexity,
                "learning_style": data.learning_style,
                "processing_time": "sync"
            }
        })

    except ValidationError as e:
        return jsonify({"error": "Invalid request data", "details": str(e)}), 400
    except Exception as e:
        logging.exception("Concept breakdown error")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/breakdown/status/<task_id>', methods=['GET'])
def get_breakdown_status(task_id):
    """Check status of async breakdown generation"""
    try:
        task = generate_concept_breakdown_async.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {'state': task.state, 'status': 'Pending...'}
        elif task.state != 'FAILURE':
            response = {
                'state': task.state,
                'result': task.result if task.ready() else None
            }
            if task.ready() and task.result['status'] == 'SUCCESS':
                response['data'] = task.result['result']
        else:
            response = {
                'state': task.state,
                'error': str(task.info)
            }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/solve-question', methods=['POST'])
@limiter.limit("20 per minute")
def api_solve_question():
    """Enhanced question solver endpoint"""
    try:
        data = QuestionSolverRequest(**request.json)
        
        solution = AIService.solve_question(
            question=data.question,
            subject=data.subject,
            exam_level=data.exam_level,
            show_steps=data.show_steps
        )
        
        return jsonify({
            "success": True,
            "data": solution,
            "metadata": {
                "subject": data.subject,
                "exam_level": data.exam_level,
                "solved_at": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logging.exception("Question solver error")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/generate-notes', methods=['POST'])
@limiter.limit("10 per minute")
def api_generate_notes():
    """Enhanced notes generation endpoint"""
    try:
        data = StudyMaterialsRequest(**request.json)
        
        # Implementation would go here
        notes = {
            "summary": "Enhanced notes would appear here",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "important_concepts": ["Concept A", "Concept B"],
            "quiz_questions": [] if not data.include_quiz else ["Q1", "Q2"]
        }
        
        return jsonify({
            "success": True,
            "data": notes
        })
        
    except Exception as e:
        logging.exception("Notes generation error")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/quiz/generate', methods=['POST'])
@token_required
def api_generate_quiz(current_user):
    """Enhanced quiz generation endpoint"""
    try:
        data = QuizGenerationRequest(**request.json)
        
        quiz = AIService.generate_quiz(
            topic=data.topic,
            difficulty=data.difficulty,
            question_count=data.question_count,
            question_types=data.question_types
        )
        
        # Track quiz generation
        track_user_activity(
            user_id=current_user,
            activity_type='quiz_generation',
            topic=data.topic,
            metadata={
                'difficulty': data.difficulty,
                'question_count': data.question_count,
                'question_types': data.question_types
            }
        )
        
        return jsonify({
            "success": True,
            "data": quiz
        })
        
    except ValidationError as e:
        return jsonify({"error": "Invalid request data", "details": str(e)}), 400
    except Exception as e:
        logging.exception("Quiz generation error")
        return jsonify({"error": str(e)}), 500


# ----- Simple analytics helpers for progress endpoint -----
def _calculate_weak_areas(progress_records: List[UserProgress]) -> List[Dict[str, Any]]:
    """Return a list of weak areas sorted by lowest proficiency.

    Each item is a dict: {"topic": str, "score": float}
    """
    if not progress_records:
        return []

    # Consider topics with proficiency < 60 as weak
    weak = [
        {"topic": p.topic, "score": float(p.proficiency_score or 0.0)}
        for p in progress_records
        if (p.proficiency_score or 0.0) < 60
    ]
    weak.sort(key=lambda x: x["score"])
    return weak


def _calculate_strengths(progress_records: List[UserProgress]) -> List[Dict[str, Any]]:
    """Return a list of strong areas sorted by highest proficiency.

    Each item is a dict: {"topic": str, "score": float}
    """
    if not progress_records:
        return []

    strong = [
        {"topic": p.topic, "score": float(p.proficiency_score or 0.0)}
        for p in progress_records
        if (p.proficiency_score or 0.0) >= 75
    ]
    strong.sort(key=lambda x: x["score"], reverse=True)
    return strong


def _generate_study_insights(progress_records: List[UserProgress], recent_sessions: List[StudySession]) -> Dict[str, Any]:
    """Generate lightweight study insights from progress and recent sessions."""
    insights: Dict[str, Any] = {}

    weak = _calculate_weak_areas(progress_records)
    strengths = _calculate_strengths(progress_records)

    insights["focus_topics"] = [w["topic"] for w in weak[:3]]
    insights["strengths"] = [s["topic"] for s in strengths[:5]]

    # Estimate average session length and weekly recommendation
    if recent_sessions:
        avg_minutes = sum(s.duration_minutes for s in recent_sessions) / len(recent_sessions)
    else:
        avg_minutes = 0

    # Suggest weekly hours: more if weaknesses exist
    suggested_hours = 5 if weak else 3
    insights["recommended_weekly_hours"] = suggested_hours
    insights["avg_session_minutes"] = round(avg_minutes, 1)

    # Actionable tip
    tips = []
    if weak:
        tips.append("Spend short, focused sessions on your weakest topics and use active recall.")
    else:
        tips.append("Maintain regular review and practice to keep strengths sharp.")

    if avg_minutes and avg_minutes < 20:
        tips.append("Try slightly longer sessions (25-45 minutes) to improve retention.")

    insights["tips"] = tips
    return insights


@app.route('/api/v1/progress', methods=['GET'])
@token_required
def api_get_progress(current_user):
    """Enhanced user progress endpoint"""
    try:
        # Get user progress from database
        progress_records = UserProgress.query.filter_by(user_id=current_user).all()

        # Calculate overall progress
        total_score = sum(p.proficiency_score for p in progress_records)
        avg_score = total_score / len(progress_records) if progress_records else 0

        # Get recent sessions
        recent_sessions = StudySession.query.filter_by(user_id=current_user)\
            .order_by(StudySession.created_at.desc())\
            .limit(10)\
            .all()

        # Compute analytics using module-level helper functions
        weak_areas = _calculate_weak_areas(progress_records)
        strengths = _calculate_strengths(progress_records)
        study_insights = _generate_study_insights(progress_records, recent_sessions)

        return jsonify({
            "success": True,
            "data": {
                "overall_progress": avg_score,
                "subjects_studied": len(progress_records),
                "total_study_time": sum(s.duration_minutes for s in recent_sessions),
                "weak_areas": weak_areas,
                "strengths": strengths,
                "recent_activity": [
                    {
                        "topic": session.topic,
                        "activity": session.activity_type,
                        "date": session.created_at.isoformat()
                    } for session in recent_sessions
                ],
                "study_insights": study_insights
            }
        })

    except Exception as e:
        logging.exception("Progress retrieval error")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/module/generate', methods=['POST'])
def api_generate_module():
    """Generate a multi-depth learning module for a topic.

    Request JSON: { "topic": "Linear regression" }
    Returns structured JSON with keys: topic, short, medium, deep
    """
    try:
        body = request.get_json(force=True)
        topic = body.get('topic') if body else None
        if not topic:
            return jsonify({"error": "Missing 'topic' in request body"}), 400

        module = AIService.generate_multi_depth_module(topic)

        return jsonify({"success": True, "data": module})
    except Exception as e:
        logging.exception("Module generation error")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/analytics/study-patterns', methods=['GET'])
@token_required
def api_study_patterns(current_user):
    """Advanced analytics for study patterns"""
    try:
        # This would implement sophisticated analytics
        # For now, return mock data structure
        return jsonify({
            "success": True,
            "data": {
                "optimal_study_times": ["Morning", "Evening"],
                "recommended_topics": ["Calculus", "Physics"],
                "learning_velocity": 85,
                "retention_rate": 78,
                "weekly_goals": {
                    "topics_to_review": 3,
                    "practice_quizzes": 2,
                    "study_hours": 10
                }
            }
        })
    except Exception as e:
        logging.exception("Analytics error")
        return jsonify({"error": str(e)}), 500

# Utility Routes
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "unknown",
            "redis": "unknown",
            "ai_service": "unknown"
        }
    }
    
    # Check database
    try:
        db.session.execute('SELECT 1')
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        except redis.RedisError:
            health_status["services"]["redis"] = "unhealthy"
            health_status["status"] = "degraded"
    
    # Check AI service
    try:
        if Config.OPENAI_API_KEY:
            # Simple check - list models
            openai.Model.list()
            health_status["services"]["ai_service"] = "healthy"
        else:
            health_status["services"]["ai_service"] = "mock_mode"
    except Exception:
        health_status["services"]["ai_service"] = "unhealthy"
        health_status["status"] = "degraded"
    
    return jsonify(health_status)

@app.route('/api/v1/system/info', methods=['GET'])
def system_info():
    """System information endpoint"""
    return jsonify({
        "app": "ThinkWise AI Study Companion",
        "version": "1.0.0",
        "environment": "development" if app.debug else "production",
        "features": {
            "authentication": True,
            "caching": Config.ENABLE_CACHING,
            "analytics": Config.ENABLE_ANALYTICS,
            "async_processing": Config.ENABLE_QUEUE
        },
        "limits": {
            "rate_limit": Config.RATE_LIMIT,
            "ai_rate_limit": Config.AI_RATE_LIMIT,
            "max_tokens": Config.MAX_TOKENS
        }
    })

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@limiter.request_filter
def exempt_health_check():
    """Exempt health checks from rate limiting"""
    return request.endpoint == 'health_check'

# Initialization
def setup_logging():
    """Enhanced logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        # Remove request_id from the default format to avoid KeyError when not present.
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[
            logging.FileHandler('thinkwise.log'),
            logging.StreamHandler()
        ]
    )

def init_db():
    """Initialize database tables"""
    try:
        # Creating tables requires an application context for Flask-SQLAlchemy
        try:
            with app.app_context():
                Base.metadata.create_all(db.engine)
            logging.info("Database tables created successfully")
        except RuntimeError:
            # If called outside of an application context, log at debug level
            # and skip automatic creation (caller can create tables in an app context).
            logging.debug("init_db called outside application context; skipping create_all()")
    except Exception as e:
        logging.error(f"Failed to create database tables: {e}")

if __name__ == '__main__':
    setup_logging()
    init_db()
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )