"""
Comprehensive tests for CourseSearchTool functionality.
Tests both normal operations and edge cases that might cause "query failed".
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore, SearchResults
from config import config
from test_helpers import assert_search_result_format, assert_error_message_format


class TestCourseSearchTool(unittest.TestCase):
    """Test CourseSearchTool functionality comprehensively"""

    def setUp(self):
        """Set up test environment"""
        # Use real vector store for most tests
        self.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        self.search_tool = CourseSearchTool(self.vector_store)

    def test_01_tool_definition_structure(self):
        """Test that tool definition has correct structure"""
        print("\n=== Testing Tool Definition Structure ===")
        
        tool_def = self.search_tool.get_tool_definition()
        
        # Check required fields
        self.assertIn("name", tool_def)
        self.assertIn("description", tool_def)
        self.assertIn("input_schema", tool_def)
        
        # Check tool name
        self.assertEqual(tool_def["name"], "search_course_content")
        
        # Check schema structure
        schema = tool_def["input_schema"]
        self.assertIn("type", schema)
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        # Check required parameters
        self.assertEqual(schema["required"], ["query"])
        
        # Check optional parameters exist
        properties = schema["properties"]
        self.assertIn("query", properties)
        self.assertIn("course_name", properties)
        self.assertIn("lesson_number", properties)
        
        print("✅ Tool definition structure is correct")

    def test_02_basic_search_functionality(self):
        """Test basic search with various query types"""
        print("\n=== Testing Basic Search Functionality ===")
        
        test_queries = [
            "what is MCP",
            "how does retrieval work",
            "Claude computer use",
            "lesson introduction",
            "Anthropic AI models"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            result = self.search_tool.execute(query)
            
            # Should return string
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            # Should not be an error message
            error_indicators = ["error", "failed", "no relevant content found"]
            is_error = any(indicator in result.lower() for indicator in error_indicators)
            
            if is_error:
                print(f"  ⚠️  Query returned no results: {result[:100]}...")
            else:
                print(f"  ✅ Query successful, result length: {len(result)}")
                # Should have course/lesson headers
                self.assertTrue("[" in result and "]" in result, 
                              "Result should have course/lesson headers")

    def test_03_course_name_filtering(self):
        """Test course name filtering functionality"""
        print("\n=== Testing Course Name Filtering ===")
        
        # Test with exact course names
        known_courses = [
            "MCP: Build Rich-Context AI Apps with Anthropic",
            "Advanced Retrieval for AI with Chroma",
            "Building Towards Computer Use with Anthropic"
        ]
        
        for course_name in known_courses:
            print(f"\nTesting with course: '{course_name}'")
            result = self.search_tool.execute("introduction", course_name=course_name)
            
            self.assertIsInstance(result, str)
            
            if "no relevant content found" in result.lower():
                print(f"  ⚠️  No content found in course")
            else:
                print(f"  ✅ Found content in specific course")
                # Result should mention the specific course
                self.assertTrue(course_name in result or course_name[:20] in result,
                              "Result should contain the specified course name")

    def test_04_partial_course_name_matching(self):
        """Test partial course name matching"""
        print("\n=== Testing Partial Course Name Matching ===")
        
        partial_matches = [
            ("MCP", "MCP: Build Rich-Context AI Apps with Anthropic"),
            ("Retrieval", "Advanced Retrieval for AI with Chroma"),
            ("Computer Use", "Building Towards Computer Use with Anthropic"),
            ("Anthropic", "Building Towards Computer Use with Anthropic")  # Should match first
        ]
        
        for partial, expected_full in partial_matches:
            print(f"\nTesting partial match: '{partial}' -> expected: '{expected_full}'")
            result = self.search_tool.execute("lesson", course_name=partial)
            
            self.assertIsInstance(result, str)
            
            if "no course found matching" in result.lower():
                print(f"  ❌ Course resolution failed")
                self.fail(f"Course resolution failed for '{partial}'")
            elif "no relevant content found" in result.lower():
                print(f"  ⚠️  Course resolved but no content found")
            else:
                print(f"  ✅ Course resolved and content found")

    def test_05_lesson_number_filtering(self):
        """Test lesson number filtering"""
        print("\n=== Testing Lesson Number Filtering ===")
        
        # Test various lesson numbers
        for lesson_num in [1, 2, 3, 5, 8]:
            print(f"\nTesting lesson {lesson_num}")
            result = self.search_tool.execute("introduction", lesson_number=lesson_num)
            
            self.assertIsInstance(result, str)
            
            if "no relevant content found" in result.lower():
                print(f"  ⚠️  No content found for lesson {lesson_num}")
            else:
                print(f"  ✅ Found content for lesson {lesson_num}")
                # Should mention the lesson number
                self.assertTrue(f"Lesson {lesson_num}" in result,
                              f"Result should mention lesson {lesson_num}")

    def test_06_combined_filtering(self):
        """Test combined course name and lesson number filtering"""
        print("\n=== Testing Combined Filtering ===")
        
        test_cases = [
            ("MCP", 1, "what is MCP"),
            ("Retrieval", 2, "embeddings"),
            ("Computer Use", 1, "overview")
        ]
        
        for course_partial, lesson_num, query in test_cases:
            print(f"\nTesting: course='{course_partial}', lesson={lesson_num}, query='{query}'")
            result = self.search_tool.execute(query, course_name=course_partial, lesson_number=lesson_num)
            
            self.assertIsInstance(result, str)
            
            if "no course found" in result.lower():
                print(f"  ❌ Course '{course_partial}' not found")
            elif "no relevant content found" in result.lower():
                print(f"  ⚠️  No content found for specific course/lesson combination")
            else:
                print(f"  ✅ Found content for specific course/lesson combination")
                # Should mention both course and lesson
                self.assertTrue(f"Lesson {lesson_num}" in result)

    def test_07_edge_case_queries(self):
        """Test edge case queries that might cause issues"""
        print("\n=== Testing Edge Case Queries ===")
        
        edge_cases = [
            "",  # Empty query
            "a",  # Single character
            "123",  # Numbers only
            "!@#$%",  # Special characters
            "x" * 1000,  # Very long query
            "query with\nnewlines\nand\ttabs",  # Newlines and tabs
            "query with 'quotes' and \"double quotes\"",  # Quotes
        ]
        
        for query in edge_cases:
            print(f"\nTesting edge case: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            try:
                result = self.search_tool.execute(query)
                self.assertIsInstance(result, str)
                print(f"  ✅ Handled edge case successfully")
                
                # Empty query should probably return an error or empty result
                if query == "":
                    print(f"  Result for empty query: {result[:100]}")
                    
            except Exception as e:
                print(f"  ❌ Edge case caused exception: {e}")
                self.fail(f"Edge case query '{query[:50]}' caused exception: {e}")

    def test_08_invalid_course_names(self):
        """Test handling of invalid course names"""
        print("\n=== Testing Invalid Course Names ===")
        
        invalid_courses = [
            "NonexistentCourse",
            "123456",
            "",
            "Course That Doesn't Exist At All",
            "!@#$%^&*()"
        ]
        
        for invalid_course in invalid_courses:
            print(f"\nTesting invalid course: '{invalid_course}'")
            result = self.search_tool.execute("test query", course_name=invalid_course)
            
            self.assertIsInstance(result, str)
            # Should indicate course not found
            self.assertTrue("no course found matching" in result.lower() or 
                           "no relevant content found" in result.lower(),
                           f"Should indicate course not found for '{invalid_course}'")
            print(f"  ✅ Properly handled invalid course name")

    def test_09_sources_tracking(self):
        """Test that sources are properly tracked"""
        print("\n=== Testing Sources Tracking ===")
        
        # Execute a search that should return results
        result = self.search_tool.execute("MCP introduction")
        
        # Check that sources were tracked
        sources = self.search_tool.last_sources
        print(f"Sources found: {len(sources)}")
        
        if sources:
            print("✅ Sources were tracked")
            # Check source structure
            for i, source in enumerate(sources[:3]):  # Check first 3
                print(f"  Source {i+1}: {source}")
                self.assertIsInstance(source, dict)
                self.assertIn("text", source)
                # Link is optional
                if "link" in source:
                    self.assertIsInstance(source["link"], str)
        else:
            print("⚠️  No sources tracked (this might be normal if no results found)")

    def test_10_mock_error_conditions(self):
        """Test behavior under error conditions using mocks"""
        print("\n=== Testing Error Conditions ===")
        
        # Test with mock vector store that returns errors
        mock_store = Mock()
        error_tool = CourseSearchTool(mock_store)
        
        # Test search error
        mock_store.search.return_value = SearchResults.empty("Database connection failed")
        result = error_tool.execute("test query")
        self.assertEqual(result, "Database connection failed")
        print("✅ Handled database error correctly")
        
        # Test exception during search
        mock_store.search.side_effect = Exception("Unexpected error")
        try:
            result = error_tool.execute("test query")
            # Should handle exception gracefully
            print(f"  Exception result: {result}")
        except Exception as e:
            print(f"❌ Exception not handled: {e}")
            self.fail("Tool should handle search exceptions gracefully")

    def test_11_tool_manager_integration(self):
        """Test integration with ToolManager"""
        print("\n=== Testing ToolManager Integration ===")
        
        # Create tool manager and register tool
        tool_manager = ToolManager()
        tool_manager.register_tool(self.search_tool)
        
        # Test tool definitions
        definitions = tool_manager.get_tool_definitions()
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "search_course_content")
        print("✅ Tool properly registered with ToolManager")
        
        # Test tool execution through manager
        result = tool_manager.execute_tool("search_course_content", query="MCP basics")
        self.assertIsInstance(result, str)
        print("✅ Tool executed successfully through ToolManager")
        
        # Test source retrieval
        sources = tool_manager.get_last_sources()
        print(f"Sources retrieved through manager: {len(sources)}")
        
        # Test unknown tool
        unknown_result = tool_manager.execute_tool("unknown_tool", query="test")
        self.assertEqual(unknown_result, "Tool 'unknown_tool' not found")
        print("✅ Unknown tool handled correctly")


def run_course_search_tool_tests():
    """Run CourseSearchTool tests with detailed output"""
    print("=" * 70)
    print("COURSE SEARCH TOOL COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCourseSearchTool)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("COURSE SEARCH TOOL TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}")
            print(f"  {trace.split('AssertionError:')[-1].strip() if 'AssertionError:' in trace else trace}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"- {test}")
            print(f"  {trace.split('Exception:')[-1].strip() if 'Exception:' in trace else trace}")
    
    return result


if __name__ == "__main__":
    run_course_search_tool_tests()