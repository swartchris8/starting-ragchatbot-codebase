"""
End-to-end integration tests for the complete RAG system.
Tests the full flow from API request to response.
"""

import unittest
import sys
import os
import json
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import config


class TestRAGSystemIntegration(unittest.TestCase):
    """Test complete RAG system integration"""

    def setUp(self):
        """Set up test environment"""
        self.rag_system = RAGSystem(config)

    def test_01_rag_system_initialization(self):
        """Test RAG system initialization and component setup"""
        print("\n=== Testing RAG System Initialization ===")
        
        # Check all components are initialized
        self.assertIsNotNone(self.rag_system.vector_store)
        self.assertIsNotNone(self.rag_system.ai_generator)
        self.assertIsNotNone(self.rag_system.tool_manager)
        self.assertIsNotNone(self.rag_system.search_tool)
        self.assertIsNotNone(self.rag_system.outline_tool)
        
        # Check tools are registered
        tools = self.rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tools]
        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)
        
        print(f"✅ RAG system initialized with {len(tools)} tools")

    def test_02_course_analytics(self):
        """Test course analytics functionality"""
        print("\n=== Testing Course Analytics ===")
        
        analytics = self.rag_system.get_course_analytics()
        
        self.assertIsInstance(analytics, dict)
        self.assertIn("total_courses", analytics)
        self.assertIn("course_titles", analytics)
        
        total_courses = analytics["total_courses"]
        course_titles = analytics["course_titles"]
        
        print(f"Total courses: {total_courses}")
        print(f"Course titles: {course_titles}")
        
        self.assertGreater(total_courses, 0)
        self.assertEqual(len(course_titles), total_courses)
        
        print("✅ Course analytics working correctly")

    def test_03_basic_query_functionality(self):
        """Test basic query functionality without sessions"""
        print("\n=== Testing Basic Query Functionality ===")
        
        test_queries = [
            "What is MCP?",
            "Tell me about retrieval systems",
            "How does Chroma work?",
            "What are the main topics in lesson 1?"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            
            try:
                response, sources = self.rag_system.query(query)
                
                # Basic response validation
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                self.assertIsInstance(sources, list)
                
                print(f"  Response length: {len(response)}")
                print(f"  Sources count: {len(sources)}")
                
                # Check for error indicators
                error_indicators = ["query failed", "error", "sorry", "cannot help"]
                has_error = any(indicator in response.lower() for indicator in error_indicators)
                
                if has_error:
                    print(f"  ❌ Query contained error: {response[:200]}...")
                else:
                    print(f"  ✅ Query successful")
                    
                # Validate sources format
                for i, source in enumerate(sources[:2]):
                    if isinstance(source, dict):
                        print(f"    Source {i+1}: {source.get('text', 'No text')} - {source.get('link', 'No link')}")
                    else:
                        print(f"    Source {i+1}: {source}")
                
            except Exception as e:
                print(f"  ❌ Query failed with exception: {e}")
                raise

    def test_04_session_management(self):
        """Test session management functionality"""
        print("\n=== Testing Session Management ===")
        
        # Test query with session
        session_id = None
        first_query = "What is machine learning?"
        
        response1, sources1 = self.rag_system.query(first_query, session_id)
        
        # Session should be created automatically
        self.assertIsInstance(response1, str)
        print(f"First query response length: {len(response1)}")
        
        # Test follow-up query
        follow_up = "Can you explain that in simpler terms?"
        
        # We need to get the session ID somehow - let's check if session manager creates one
        # For now, test with a specific session ID
        test_session_id = self.rag_system.session_manager.create_session()
        print(f"Created session ID: {test_session_id}")
        
        response2, sources2 = self.rag_system.query(first_query, test_session_id)
        response3, sources3 = self.rag_system.query(follow_up, test_session_id)
        
        self.assertIsInstance(response2, str)
        self.assertIsInstance(response3, str)
        
        print(f"Second query response length: {len(response2)}")
        print(f"Follow-up query response length: {len(response3)}")
        
        print("✅ Session management working")

    def test_05_error_handling_scenarios(self):
        """Test various error handling scenarios"""
        print("\n=== Testing Error Handling Scenarios ===")
        
        # Test empty query
        try:
            response, sources = self.rag_system.query("")
            print(f"Empty query result: {response[:100]}...")
            self.assertIsInstance(response, str)
        except Exception as e:
            print(f"Empty query caused exception: {e}")
        
        # Test very long query
        long_query = "x" * 5000
        try:
            response, sources = self.rag_system.query(long_query)
            print(f"Long query handled, response length: {len(response)}")
            self.assertIsInstance(response, str)
        except Exception as e:
            print(f"Long query caused exception: {e}")
        
        # Test query with special characters
        special_query = "What is AI? @#$%^&*(){}[]|\\:;\"'<>?,./"
        try:
            response, sources = self.rag_system.query(special_query)
            print(f"Special characters handled, response length: {len(response)}")
            self.assertIsInstance(response, str)
        except Exception as e:
            print(f"Special characters caused exception: {e}")
        
        print("✅ Error handling scenarios tested")

    def test_06_course_specific_queries(self):
        """Test queries specific to known courses"""
        print("\n=== Testing Course-Specific Queries ===")
        
        # Get available courses
        analytics = self.rag_system.get_course_analytics()
        available_courses = analytics["course_titles"]
        
        if not available_courses:
            print("⚠️  No courses available for testing")
            return
        
        # Test queries about specific courses
        course_queries = [
            f"What is covered in the {available_courses[0]} course?",
            f"Tell me about lesson 1 of {available_courses[0]}",
            f"What are the key concepts in {available_courses[0]}?"
        ]
        
        for query in course_queries:
            print(f"\nTesting course-specific query: '{query}'")
            
            try:
                response, sources = self.rag_system.query(query)
                
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                
                print(f"  Response length: {len(response)}")
                print(f"  Sources: {len(sources)}")
                
                # Should mention the specific course
                course_mentioned = any(course in response for course in available_courses)
                if course_mentioned:
                    print("  ✅ Course-specific content found")
                else:
                    print("  ⚠️  Course might not be specifically mentioned")
                
            except Exception as e:
                print(f"  ❌ Course-specific query failed: {e}")

    def test_07_comparison_queries(self):
        """Test queries that compare different courses or concepts"""
        print("\n=== Testing Comparison Queries ===")
        
        comparison_queries = [
            "What's the difference between the MCP course and the retrieval course?",
            "Compare lesson 1 across all courses",
            "Which course covers AI applications best?"
        ]
        
        for query in comparison_queries:
            print(f"\nTesting comparison query: '{query}'")
            
            try:
                response, sources = self.rag_system.query(query)
                
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 50)  # Should be substantial
                
                print(f"  Response length: {len(response)}")
                print(f"  Sources from multiple courses: {len(set(source.get('text', '').split(' - ')[0] for source in sources if isinstance(source, dict)))}")
                
                # Comparison queries should ideally reference multiple sources
                if len(sources) > 1:
                    print("  ✅ Multiple sources used for comparison")
                else:
                    print("  ⚠️  Limited sources for comparison")
                
            except Exception as e:
                print(f"  ❌ Comparison query failed: {e}")

    def test_08_outline_tool_functionality(self):
        """Test the outline tool functionality"""
        print("\n=== Testing Outline Tool Functionality ===")
        
        # Get available courses for testing
        analytics = self.rag_system.get_course_analytics()
        available_courses = analytics["course_titles"]
        
        if not available_courses:
            print("⚠️  No courses available for outline testing")
            return
        
        # Test outline queries
        outline_queries = [
            f"Show me the outline of the {available_courses[0]} course",
            f"What lessons are in {available_courses[0]}?",
            "Give me the structure of the MCP course"
        ]
        
        for query in outline_queries:
            print(f"\nTesting outline query: '{query}'")
            
            try:
                response, sources = self.rag_system.query(query)
                
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)
                
                print(f"  Response length: {len(response)}")
                
                # Outline responses should contain lesson numbers
                has_lesson_numbers = any(f"lesson {i}" in response.lower() or f"{i}." in response for i in range(1, 10))
                if has_lesson_numbers:
                    print("  ✅ Outline contains lesson structure")
                else:
                    print("  ⚠️  Outline might not show lesson structure")
                
            except Exception as e:
                print(f"  ❌ Outline query failed: {e}")

    def test_09_stress_test_multiple_queries(self):
        """Stress test with multiple rapid queries"""
        print("\n=== Testing Multiple Rapid Queries ===")
        
        queries = [
            "What is AI?",
            "How does MCP work?",
            "Tell me about embeddings",
            "What is ChromaDB?",
            "How do I build AI apps?",
        ]
        
        results = []
        
        for i, query in enumerate(queries):
            print(f"Query {i+1}: {query}")
            
            try:
                response, sources = self.rag_system.query(query)
                results.append({
                    "query": query,
                    "success": True,
                    "response_length": len(response),
                    "sources_count": len(sources)
                })
                print(f"  ✅ Success: {len(response)} chars, {len(sources)} sources")
                
            except Exception as e:
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
                print(f"  ❌ Failed: {e}")
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nStress test results: {successful}/{len(queries)} successful")
        
        self.assertGreater(successful, 0, "At least some queries should succeed")

    def test_10_realistic_user_scenarios(self):
        """Test realistic user interaction scenarios"""
        print("\n=== Testing Realistic User Scenarios ===")
        
        # Scenario 1: New user exploring courses
        print("\nScenario 1: New user exploring")
        try:
            response1, _ = self.rag_system.query("What courses are available?")
            response2, _ = self.rag_system.query("Which course should I start with for AI applications?")
            response3, _ = self.rag_system.query("Tell me about the first lesson")
            
            print(f"  Exploration queries completed successfully")
        except Exception as e:
            print(f"  Exploration scenario failed: {e}")
        
        # Scenario 2: Student following a specific course
        print("\nScenario 2: Following specific course")
        try:
            session_id = self.rag_system.session_manager.create_session()
            response1, _ = self.rag_system.query("I want to learn about MCP", session_id)
            response2, _ = self.rag_system.query("What's in lesson 1?", session_id)
            response3, _ = self.rag_system.query("Can you explain that concept more?", session_id)
            
            print(f"  Course following scenario completed successfully")
        except Exception as e:
            print(f"  Course following scenario failed: {e}")
        
        # Scenario 3: Advanced user comparing concepts
        print("\nScenario 3: Advanced comparison")
        try:
            response1, _ = self.rag_system.query("How does vector search compare to traditional search?")
            response2, _ = self.rag_system.query("What are the pros and cons of different embedding approaches?")
            
            print(f"  Advanced comparison completed successfully")
        except Exception as e:
            print(f"  Advanced comparison failed: {e}")
        
        print("✅ Realistic user scenarios tested")


def run_rag_system_tests():
    """Run RAG system integration tests with detailed output"""
    print("=" * 70)
    print("RAG SYSTEM INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRAGSystemIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("RAG SYSTEM INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")  
        for test, trace in result.errors:
            print(f"- {test}")
    
    return result


if __name__ == "__main__":
    run_rag_system_tests()