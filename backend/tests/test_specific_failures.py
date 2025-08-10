"""
Focused tests to identify specific failure scenarios causing "query failed".
Based on the "list index out of range" error found in comparison queries.
"""

import os
import sys
import traceback
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from config import config
from rag_system import RAGSystem
from search_tools import CourseSearchTool
from vector_store import VectorStore


class TestSpecificFailures(unittest.TestCase):
    """Test specific failure scenarios"""

    def setUp(self):
        """Set up test environment"""
        self.rag_system = RAGSystem(config)

    def test_01_problematic_comparison_queries(self):
        """Test the specific queries that cause list index out of range"""
        print("\n=== Testing Problematic Comparison Queries ===")

        problematic_queries = [
            "What's the difference between the MCP course and the retrieval course?",
            "Compare MCP and Chroma courses",
            "What are the differences between lesson 1 in different courses?",
            "How do the courses compare to each other?",
        ]

        for query in problematic_queries:
            print(f"\nTesting problematic query: '{query}'")

            try:
                response, sources = self.rag_system.query(query)
                print(
                    f"  ✅ Query succeeded: {len(response)} chars, {len(sources)} sources"
                )

            except Exception as e:
                print(f"  ❌ Query failed with exception: {type(e).__name__}: {e}")
                print(f"  Full traceback:")
                traceback.print_exc()

                # This is expected - we want to capture the exact error
                self.fail(f"Query '{query}' caused: {e}")

    def test_02_ai_generator_tool_response_handling(self):
        """Test AIGenerator's handling of tool responses for edge cases"""
        print("\n=== Testing AI Generator Tool Response Handling ===")

        # Test with real tools but capture any list index errors
        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        from search_tools import CourseSearchTool, ToolManager

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        tool_manager.register_tool(search_tool)

        # Test queries that might cause tool response handling issues
        edge_case_queries = [
            "Compare all courses and tell me which is best",
            "What's different between course A and course B?",
            "Find similarities and differences across lessons",
        ]

        for query in edge_case_queries:
            print(f"\nTesting edge case: '{query}'")

            try:
                response = ai_gen.generate_response(
                    query=query,
                    tools=tool_manager.get_tool_definitions(),
                    tool_manager=tool_manager,
                )
                print(f"  ✅ AIGenerator handled successfully: {len(response)} chars")

            except IndexError as e:
                print(f"  ❌ IndexError in AIGenerator: {e}")
                print("  Full traceback:")
                traceback.print_exc()

                # This helps us identify where the list index error occurs

            except Exception as e:
                print(f"  ❌ Other exception: {type(e).__name__}: {e}")

    def test_03_tool_response_format_analysis(self):
        """Analyze tool response formats that might cause parsing issues"""
        print("\n=== Analyzing Tool Response Formats ===")

        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        search_tool = CourseSearchTool(vector_store)

        # Test various search scenarios
        test_scenarios = [
            ("basic", "What is MCP?"),
            (
                "course_filter",
                "introduction",
                "MCP: Build Rich-Context AI Apps with Anthropic",
            ),
            ("lesson_filter", "overview", None, 1),
            ("combined_filter", "basics", "Advanced Retrieval", 1),
            ("no_results", "xyznonexistentquery123"),
            ("empty_query", ""),
        ]

        for scenario_name, query, course_name, lesson_number in [
            (s[0], s[1], s[2] if len(s) > 2 else None, s[3] if len(s) > 3 else None)
            for s in test_scenarios
        ]:
            print(
                f"\nTesting scenario '{scenario_name}': query='{query}', course='{course_name}', lesson={lesson_number}"
            )

            try:
                kwargs = {"query": query}
                if course_name:
                    kwargs["course_name"] = course_name
                if lesson_number:
                    kwargs["lesson_number"] = lesson_number

                result = search_tool.execute(**kwargs)

                print(f"  Result type: {type(result)}")
                print(
                    f"  Result length: {len(result) if isinstance(result, str) else 'N/A'}"
                )
                print(f"  Sources tracked: {len(search_tool.last_sources)}")

                # Analyze result format
                if isinstance(result, str):
                    has_brackets = "[" in result and "]" in result
                    has_newlines = "\n" in result
                    print(f"  Has course brackets: {has_brackets}")
                    print(f"  Has newlines: {has_newlines}")

                    if (
                        not has_brackets
                        and not "no relevant content" in result.lower()
                        and not "error" in result.lower()
                    ):
                        print(f"  ⚠️  Unusual format - no course brackets")

            except Exception as e:
                print(f"  ❌ Tool execution failed: {e}")
                traceback.print_exc()

    def test_04_mock_malformed_responses(self):
        """Test handling of various malformed API responses"""
        print("\n=== Testing Malformed Response Handling ===")

        from unittest.mock import Mock, patch

        # Test different malformed response scenarios
        malformed_scenarios = [
            ("empty_content", Mock(content=[], stop_reason="tool_use")),
            (
                "missing_text",
                Mock(content=[Mock(type="text")], stop_reason="end_turn"),
            ),  # Missing text attribute
            (
                "missing_tool_fields",
                Mock(
                    content=[Mock(type="tool_use", name="test")], stop_reason="tool_use"
                ),
            ),  # Missing id, input
        ]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            ai_gen = AIGenerator("test-key", "test-model")

            for scenario_name, malformed_response in malformed_scenarios:
                print(f"\nTesting malformed scenario: {scenario_name}")

                mock_client.messages.create.return_value = malformed_response

                try:
                    result = ai_gen.generate_response("test query")
                    print(f"  Handled gracefully: {result}")

                except IndexError as e:
                    print(f"  ❌ IndexError: {e}")
                    print(
                        f"  This is likely the source of 'list index out of range' errors"
                    )

                except Exception as e:
                    print(f"  Other exception: {type(e).__name__}: {e}")

    def test_05_session_manager_edge_cases(self):
        """Test session manager edge cases"""
        print("\n=== Testing Session Manager Edge Cases ===")

        session_manager = self.rag_system.session_manager

        # Test with invalid session IDs
        invalid_sessions = [
            None,
            "",
            "nonexistent_session",
            "session with spaces",
            "session@#$%^&*()",
        ]

        for invalid_session in invalid_sessions:
            print(f"\nTesting invalid session: '{invalid_session}'")

            try:
                history = session_manager.get_conversation_history(invalid_session)
                print(f"  History retrieved: {history}")

                # Try adding to invalid session
                session_manager.add_exchange(
                    invalid_session, "test query", "test response"
                )
                print(f"  Exchange added successfully")

            except Exception as e:
                print(f"  Exception with invalid session: {e}")

    def test_06_reproduce_query_failed_scenarios(self):
        """Try to reproduce specific 'query failed' scenarios"""
        print("\n=== Reproducing 'Query Failed' Scenarios ===")

        # Common scenarios that might lead to "query failed"
        potentially_failing_queries = [
            # Complex comparison queries
            "Compare all the courses and tell me which one I should take first",
            "What are the main differences between MCP and retrieval approaches?",
            "How do lesson 1 topics differ across courses?",
            # Queries that might return no results
            "Tell me about lesson 50",  # Probably doesn't exist
            "What is quantum computing in these courses?",  # Probably not covered
            # Complex filtering
            "Find content about machine learning in lesson 5 of the Chroma course",
            "What does lesson 3 say about API integration?",
            # Edge cases
            "",  # Empty
            "a",  # Too short
            " " * 100,  # Just spaces
        ]

        for query in potentially_failing_queries:
            print(
                f"\nTesting potentially failing query: '{query[:50]}{'...' if len(query) > 50 else ''}'"
            )

            try:
                response, sources = self.rag_system.query(query)

                # Check for "query failed" or similar indicators
                if "query failed" in response.lower():
                    print(f"  ❌ Found 'query failed' in response!")
                    print(f"  Response: {response}")
                elif any(
                    indicator in response.lower()
                    for indicator in ["error", "sorry", "cannot", "unable"]
                ):
                    print(f"  ⚠️  Response contains error indicators")
                    print(f"  Response: {response[:200]}...")
                else:
                    print(f"  ✅ Query handled successfully: {len(response)} chars")

            except Exception as e:
                print(f"  ❌ Query raised exception: {type(e).__name__}: {e}")

                # Check if this is the "list index out of range" error
                if "list index out of range" in str(e):
                    print(f"  🎯 Found the 'list index out of range' error!")
                    traceback.print_exc()


def run_specific_failure_tests():
    """Run specific failure tests with detailed output"""
    print("=" * 70)
    print("SPECIFIC FAILURE ANALYSIS TEST SUITE")
    print("=" * 70)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpecificFailures)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("SPECIFIC FAILURE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result


if __name__ == "__main__":
    run_specific_failure_tests()
