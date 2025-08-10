import os
import shutil
import sys
import tempfile
import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def test_config():
    """Test configuration with temporary paths"""
    return Config(
        ANTHROPIC_API_KEY="test-key-12345",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH="./test_chroma_db",
    )


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is ML?",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Supervised Learning",
                lesson_link="https://example.com/lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Unsupervised Learning",
                lesson_link="https://example.com/lesson3",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine Learning is a subset of artificial intelligence that focuses on algorithms.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Supervised learning uses labeled data to train models for prediction tasks.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Unsupervised learning finds patterns in data without labeled examples.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_search_results():
    """Mock search results for testing"""
    return SearchResults(
        documents=[
            "Machine Learning is a subset of artificial intelligence that focuses on algorithms.",
            "Supervised learning uses labeled data to train models for prediction tasks.",
        ],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Database connection error")


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(
        documents=["Test document content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()

    # Mock a simple text response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"

    # Mock a tool use response
    mock_tool_response = Mock()
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.id = "tool_123"
    mock_tool_content.input = {"query": "test query"}
    mock_tool_response.content = [mock_tool_content]
    mock_tool_response.stop_reason = "tool_use"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def temp_chroma_db():
    """Temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_mock_chroma_results(
    documents: List[str], metadata: List[Dict[str, Any]]
) -> Dict:
    """Helper function to create mock ChromaDB results"""
    return {
        "documents": [documents],
        "metadatas": [metadata],
        "distances": [[0.1] * len(documents)] if documents else [[]],
    }


# API Testing Fixtures

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a test response from the RAG system.",
        [{"text": "Test source content", "link": "https://example.com/lesson1"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Machine Learning", "Advanced AI Applications"]
    }
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    return mock_rag

@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file dependencies"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict
    
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware for testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models (copy from main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, Dict[str, str]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints with mocked RAG system
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "rag-system"}
    
    return app

@pytest.fixture
def client(test_app):
    """Test client for API endpoints"""
    return TestClient(test_app)

@pytest.fixture
async def async_client(test_app):
    """Async test client for API endpoints"""
    from httpx import ASGITransport
    async with httpx.AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_query_request():
    """Sample query request payload"""
    return {
        "query": "What is machine learning?",
        "session_id": None
    }

@pytest.fixture  
def sample_query_request_with_session():
    """Sample query request with session ID"""
    return {
        "query": "Tell me more about that topic",
        "session_id": "test-session-123"
    }

@pytest.fixture
def invalid_query_request():
    """Invalid query request for error testing"""
    return {
        "query": "",  # Empty query
        "session_id": "invalid-session"
    }

@pytest.fixture
def expected_query_response():
    """Expected structure of query response"""
    return {
        "answer": str,
        "sources": list,
        "session_id": str
    }

@pytest.fixture
def expected_course_stats():
    """Expected structure of course stats response"""
    return {
        "total_courses": int,
        "course_titles": list
    }

@pytest.fixture
def mock_rag_system_error():
    """Mock RAG system that raises errors for testing"""
    mock_rag = Mock()
    mock_rag.query.side_effect = Exception("RAG system error")
    mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
    return mock_rag

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
