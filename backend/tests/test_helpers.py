"""Test helper utilities for RAG system testing"""

import os
import sys
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def assert_search_result_format(result: str, expected_course: str = None, expected_lesson: int = None):
    """Assert that search result has proper format with course/lesson headers"""
    assert isinstance(result, str)
    assert len(result) > 0
    
    if expected_course:
        assert f"[{expected_course}" in result
    
    if expected_lesson is not None:
        assert f"Lesson {expected_lesson}" in result

def assert_error_message_format(result: str):
    """Assert that error message has proper format"""
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain some indication of error or failure
    error_indicators = ["error", "failed", "not found", "no relevant content"]
    assert any(indicator.lower() in result.lower() for indicator in error_indicators)

def create_test_environment_with_real_data():
    """Create a test environment with real data loaded"""
    from config import config
    from rag_system import RAGSystem
    from vector_store import VectorStore
    
    # Initialize with real config but test database
    test_config = config
    test_config.CHROMA_PATH = "./test_real_chroma_db"
    
    # Create RAG system
    rag = RAGSystem(test_config)
    
    # Load real documents if available
    docs_path = "../docs"
    if os.path.exists(docs_path):
        rag.add_course_folder(docs_path, clear_existing=True)
    
    return rag

def mock_anthropic_api_responses():
    """Create mock responses for different Anthropic API scenarios"""
    
    # Standard text response
    text_response = Mock()
    text_response.content = [Mock(text="This is a standard text response")]
    text_response.stop_reason = "end_turn"
    
    # Tool use response
    tool_response = Mock()
    tool_content = Mock()
    tool_content.type = "tool_use"
    tool_content.name = "search_course_content"
    tool_content.id = "tool_12345"
    tool_content.input = {"query": "machine learning"}
    tool_response.content = [tool_content]
    tool_response.stop_reason = "tool_use"
    
    # Final response after tool use
    final_response = Mock()
    final_response.content = [Mock(text="Based on the search results, machine learning is...")]
    final_response.stop_reason = "end_turn"
    
    return {
        "text_response": text_response,
        "tool_response": tool_response,
        "final_response": final_response
    }

def simulate_query_failure_scenarios():
    """Create different failure scenario mocks"""
    
    scenarios = {}
    
    # API Key Error
    api_error = Exception("Invalid API key")
    scenarios["api_key_error"] = api_error
    
    # ChromaDB Connection Error
    chroma_error = Exception("Unable to connect to ChromaDB")
    scenarios["chroma_error"] = chroma_error
    
    # Tool Execution Error
    tool_error = Exception("Tool execution failed")
    scenarios["tool_error"] = tool_error
    
    # Empty Database Error
    empty_db_error = Exception("No data found in database")
    scenarios["empty_db_error"] = empty_db_error
    
    return scenarios

def check_chroma_db_state(chroma_path: str) -> Dict[str, Any]:
    """Check the actual state of ChromaDB for debugging"""
    import chromadb
    from chromadb.config import Settings
    
    try:
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Check collections
        collections_info = {}
        
        try:
            course_catalog = client.get_collection("course_catalog")
            catalog_count = course_catalog.count()
            collections_info["course_catalog"] = {
                "exists": True,
                "count": catalog_count,
                "sample_ids": course_catalog.get(limit=5).get("ids", [])
            }
        except Exception as e:
            collections_info["course_catalog"] = {
                "exists": False,
                "error": str(e)
            }
        
        try:
            course_content = client.get_collection("course_content")
            content_count = course_content.count()
            collections_info["course_content"] = {
                "exists": True,
                "count": content_count,
                "sample_ids": course_content.get(limit=5).get("ids", [])
            }
        except Exception as e:
            collections_info["course_content"] = {
                "exists": False,
                "error": str(e)
            }
        
        return {
            "chroma_path": chroma_path,
            "chroma_accessible": True,
            "collections": collections_info
        }
        
    except Exception as e:
        return {
            "chroma_path": chroma_path,
            "chroma_accessible": False,
            "error": str(e)
        }

def test_actual_api_connectivity(api_key: str) -> Dict[str, Any]:
    """Test actual connectivity to Anthropic API"""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Try a simple API call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        return {
            "api_accessible": True,
            "response_received": True,
            "response_content": response.content[0].text if response.content else "No content"
        }
        
    except Exception as e:
        return {
            "api_accessible": False,
            "error": str(e),
            "error_type": type(e).__name__
        }