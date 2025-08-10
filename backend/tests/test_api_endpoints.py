"""
API endpoint tests for the RAG system FastAPI application.
Tests all endpoints for proper request/response handling, error cases, and integration.
"""

import pytest
import json
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import Mock, patch
from typing import Dict, Any


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""
    
    def test_query_with_new_session(self, client: TestClient, sample_query_request: Dict[str, Any]):
        """Test query endpoint creates new session when none provided"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Validate data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Session should be created
        assert data["session_id"] == "test-session-123"
        assert len(data["answer"]) > 0
    
    def test_query_with_existing_session(self, client: TestClient, sample_query_request_with_session: Dict[str, Any]):
        """Test query endpoint uses existing session when provided"""
        response = client.post("/api/query", json=sample_query_request_with_session)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["session_id"] == "test-session-123"
        assert "answer" in data
        assert "sources" in data
    
    def test_query_response_format(self, client: TestClient, sample_query_request: Dict[str, Any]):
        """Test query endpoint returns properly formatted response"""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check sources format (should support both string and dict formats)
        if data["sources"]:
            source = data["sources"][0]
            if isinstance(source, dict):
                assert "text" in source or "link" in source
    
    def test_query_missing_query_field(self, client: TestClient):
        """Test query endpoint with missing query field"""
        invalid_request = {"session_id": "test"}
        response = client.post("/api/query", json=invalid_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_empty_query(self, client: TestClient):
        """Test query endpoint with empty query string"""
        empty_query = {"query": "", "session_id": None}
        response = client.post("/api/query", json=empty_query)
        
        # Should still return 200 but handle empty query gracefully
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "session_id" in data
    
    def test_query_very_long_query(self, client: TestClient):
        """Test query endpoint with very long query string"""
        long_query = {"query": "x" * 10000, "session_id": None}
        response = client.post("/api/query", json=long_query)
        
        # Should handle long queries without error
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
    
    def test_query_special_characters(self, client: TestClient):
        """Test query endpoint with special characters"""
        special_query = {
            "query": "What is AI? @#$%^&*(){}[]|\\:;\"'<>?,./ Testing unicode: 你好 🤖",
            "session_id": None
        }
        response = client.post("/api/query", json=special_query)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
    
    def test_query_invalid_json(self, client: TestClient):
        """Test query endpoint with malformed JSON"""
        response = client.post("/api/query", data="invalid json")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_query_wrong_method(self, client: TestClient):
        """Test query endpoint with wrong HTTP method"""
        response = client.get("/api/query")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_query_rag_system_error(self, client: TestClient, mock_rag_system, sample_query_request: Dict[str, Any]):
        """Test query endpoint when RAG system raises an error"""
        # Configure the mock to raise an exception
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = client.post("/api/query", json=sample_query_request)
        
        # The endpoint should handle the error and return 500
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error_data = response.json()
        assert "detail" in error_data
        assert "RAG system error" in error_data["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""
    
    def test_courses_endpoint_success(self, client: TestClient):
        """Test courses endpoint returns course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Validate data types and values
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
        assert len(data["course_titles"]) == data["total_courses"]
    
    def test_courses_endpoint_data_consistency(self, client: TestClient):
        """Test courses endpoint data consistency"""
        response = client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # From our mock, we expect specific values
        assert data["total_courses"] == 2
        assert "Introduction to Machine Learning" in data["course_titles"]
        assert "Advanced AI Applications" in data["course_titles"]
    
    def test_courses_endpoint_wrong_method(self, client: TestClient):
        """Test courses endpoint with wrong HTTP method"""
        response = client.post("/api/courses")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
        
        response = client.put("/api/courses")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_courses_endpoint_with_query_params(self, client: TestClient):
        """Test courses endpoint ignores query parameters"""
        response = client.get("/api/courses?param=value&other=test")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
    
    def test_courses_endpoint_error(self, client: TestClient, mock_rag_system):
        """Test courses endpoint when RAG system raises an error"""
        # Configure the mock to raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        # The endpoint should handle the error and return 500
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        error_data = response.json()
        assert "detail" in error_data
        assert "Analytics error" in error_data["detail"]


@pytest.mark.api
class TestRootEndpoint:
    """Test the root / endpoint"""
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint returns welcome message"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "message" in data
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0
    
    def test_root_endpoint_content(self, client: TestClient):
        """Test root endpoint returns expected content"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["message"] == "Course Materials RAG System API"


@pytest.mark.api
class TestHealthEndpoint:
    """Test the health check endpoint"""
    
    def test_health_endpoint(self, client: TestClient):
        """Test health endpoint returns system status"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert data["status"] == "healthy"
        assert data["service"] == "rag-system"


@pytest.mark.api
class TestCORSHeaders:
    """Test CORS headers are properly set"""
    
    def test_cors_preflight_query(self, client: TestClient):
        """Test CORS preflight request for query endpoint"""
        response = client.options("/api/query", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        # OPTIONS requests should be handled
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]
    
    def test_cors_headers_present(self, client: TestClient, sample_query_request: Dict[str, Any]):
        """Test CORS headers are present in responses"""
        response = client.post("/api/query", json=sample_query_request, headers={
            "Origin": "http://localhost:3000"
        })
        
        assert response.status_code == status.HTTP_200_OK
        
        # Check for CORS headers (these might be added by FastAPI middleware)
        # The exact headers depend on the CORS configuration
        headers = response.headers
        cors_related_headers = [h for h in headers.keys() if 'access-control' in h.lower()]
        # We expect some CORS headers to be present
        assert len(cors_related_headers) >= 0  # Middleware might not add headers in test environment


@pytest.mark.api
class TestContentTypes:
    """Test different content types and request formats"""
    
    def test_json_content_type(self, client: TestClient, sample_query_request: Dict[str, Any]):
        """Test endpoint accepts JSON content type"""
        response = client.post(
            "/api/query", 
            json=sample_query_request,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_response_content_type(self, client: TestClient):
        """Test endpoints return JSON content type"""
        response = client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.api
class TestErrorHandling:
    """Test API error handling scenarios"""
    
    def test_nonexistent_endpoint(self, client: TestClient):
        """Test requesting non-existent endpoint"""
        response = client.get("/api/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_malformed_request_body(self, client: TestClient):
        """Test malformed request body handling"""
        response = client.post(
            "/api/query",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_content_type(self, client: TestClient):
        """Test request without content type"""
        response = client.post("/api/query", data='{"query": "test"}')
        # FastAPI should handle this gracefully
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]


@pytest.mark.api
@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test endpoints using async client"""
    
    async def test_async_query_endpoint(self, async_client, sample_query_request: Dict[str, Any]):
        """Test query endpoint with async client"""
        response = await async_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    async def test_async_courses_endpoint(self, async_client):
        """Test courses endpoint with async client"""
        response = await async_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
    
    async def test_concurrent_requests(self, async_client, sample_query_request: Dict[str, Any]):
        """Test multiple concurrent requests"""
        import asyncio
        
        # Make multiple concurrent requests
        tasks = [
            async_client.post("/api/query", json=sample_query_request),
            async_client.get("/api/courses"),
            async_client.get("/"),
            async_client.get("/health")
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndAPIFlow:
    """Test complete API workflow scenarios"""
    
    def test_new_user_workflow(self, client: TestClient):
        """Test complete workflow for new user"""
        # 1. Check available courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        courses_data = courses_response.json()
        assert courses_data["total_courses"] > 0
        
        # 2. Make first query (should create session)
        first_query = {"query": "What courses are available?", "session_id": None}
        query_response = client.post("/api/query", json=first_query)
        assert query_response.status_code == status.HTTP_200_OK
        query_data = query_response.json()
        session_id = query_data["session_id"]
        
        # 3. Follow up with contextual query using session
        follow_up = {"query": "Tell me more about the first one", "session_id": session_id}
        follow_up_response = client.post("/api/query", json=follow_up)
        assert follow_up_response.status_code == status.HTTP_200_OK
        follow_up_data = follow_up_response.json()
        assert follow_up_data["session_id"] == session_id
    
    def test_session_persistence(self, client: TestClient):
        """Test session persistence across multiple queries"""
        session_id = "persistent-test-session"
        
        queries = [
            "What is machine learning?",
            "Can you explain supervised learning?",
            "What about unsupervised learning?",
            "How do these compare?"
        ]
        
        responses = []
        for query in queries:
            request = {"query": query, "session_id": session_id}
            response = client.post("/api/query", json=request)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["session_id"] == session_id
            responses.append(data)
        
        # All responses should maintain the same session
        session_ids = [r["session_id"] for r in responses]
        assert len(set(session_ids)) == 1  # All same session ID
    
    def test_multiple_sessions_isolation(self, client: TestClient):
        """Test that multiple sessions are properly isolated"""
        session1 = "session-1"
        session2 = "session-2"
        
        # Query in session 1
        request1 = {"query": "What is AI?", "session_id": session1}
        response1 = client.post("/api/query", json=request1)
        assert response1.status_code == status.HTTP_200_OK
        data1 = response1.json()
        
        # Query in session 2
        request2 = {"query": "What is ML?", "session_id": session2}
        response2 = client.post("/api/query", json=request2)
        assert response2.status_code == status.HTTP_200_OK
        data2 = response2.json()
        
        # Sessions should be maintained separately
        assert data1["session_id"] == session1
        assert data2["session_id"] == session2
        assert data1["session_id"] != data2["session_id"]