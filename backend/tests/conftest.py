import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

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
