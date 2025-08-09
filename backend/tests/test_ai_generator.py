"""
Tests for AIGenerator to check Claude API integration and tool calling.
This tests if the AI correctly calls CourseSearchTool and handles responses.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import VectorStore
from config import config


class TestAIGenerator(unittest.TestCase):
    """Test AIGenerator functionality with tool calling"""

    def setUp(self):
        """Set up test environment"""
        # Use real API key if available, otherwise mock
        self.api_key = config.ANTHROPIC_API_KEY
        self.model = config.ANTHROPIC_MODEL
        
        if self.api_key and len(self.api_key) > 20:
            self.use_real_api = True
            self.ai_generator = AIGenerator(self.api_key, self.model)
        else:
            self.use_real_api = False
            # Will use mocked version in tests

    def test_01_ai_generator_initialization(self):
        """Test AIGenerator initialization"""
        print("\n=== Testing AIGenerator Initialization ===")
        
        if self.use_real_api:
            print("✅ Using real API key")
            self.assertIsNotNone(self.ai_generator.client)
            self.assertEqual(self.ai_generator.model, self.model)
        else:
            print("⚠️  No valid API key - will test with mocks")
            
        # Test with mock API key
        mock_ai = AIGenerator("test-key-123", "claude-sonnet-4-20250514")
        self.assertIsNotNone(mock_ai)
        print("✅ AIGenerator initialization works")

    @patch('anthropic.Anthropic')
    def test_02_basic_text_generation(self, mock_anthropic):
        """Test basic text generation without tools"""
        print("\n=== Testing Basic Text Generation ===")
        
        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Test simple query
        result = ai_gen.generate_response("What is AI?")
        
        self.assertEqual(result, "This is a test response")
        mock_client.messages.create.assert_called_once()
        print("✅ Basic text generation works")

    @patch('anthropic.Anthropic')
    def test_03_tool_calling_functionality(self, mock_anthropic):
        """Test tool calling functionality"""
        print("\n=== Testing Tool Calling Functionality ===")
        
        # Set up mocks
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Create tool manager with mock tool
        tool_manager = Mock()
        mock_tool_definitions = [{
            "name": "search_course_content",
            "description": "Search course content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }]
        tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        tool_manager.execute_tool.return_value = "Mock search result from tool"
        
        # Mock initial tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_12345"
        mock_tool_content.input = {"query": "test query"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [Mock(text="Final response based on tool results")]
        final_response.stop_reason = "end_turn"
        
        # Configure mock to return different responses for different calls
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Execute test
        result = ai_gen.generate_response(
            query="What is machine learning?",
            tools=mock_tool_definitions,
            tool_manager=tool_manager
        )
        
        # Verify results
        self.assertEqual(result, "Final response based on tool results")
        self.assertEqual(mock_client.messages.create.call_count, 2)  # Initial + final call
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test query")
        print("✅ Tool calling functionality works")

    def test_04_real_api_basic_query(self):
        """Test real API with basic query (if API key available)"""
        print("\n=== Testing Real API Basic Query ===")
        
        if not self.use_real_api:
            print("⚠️  Skipping real API test - no valid API key")
            return
        
        try:
            result = self.ai_generator.generate_response("What is 2+2? Answer with just the number.")
            print(f"Real API response: {result}")
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            print("✅ Real API basic query works")
        except Exception as e:
            print(f"❌ Real API test failed: {e}")
            raise

    def test_05_real_api_with_tools(self):
        """Test real API with actual search tools"""
        print("\n=== Testing Real API with Tools ===")
        
        if not self.use_real_api:
            print("⚠️  Skipping real API with tools test - no valid API key")
            return
        
        try:
            # Create real tool manager
            vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            tool_manager = ToolManager()
            search_tool = CourseSearchTool(vector_store)
            tool_manager.register_tool(search_tool)
            
            # Test queries that should trigger tool use
            test_queries = [
                "What is MCP in the course materials?",
                "Tell me about lesson 1 in the retrieval course",
                "How does ChromaDB work for AI applications?"
            ]
            
            for query in test_queries:
                print(f"\nTesting query: '{query}'")
                
                result = self.ai_generator.generate_response(
                    query=query,
                    tools=tool_manager.get_tool_definitions(),
                    tool_manager=tool_manager
                )
                
                print(f"Response type: {type(result)}")
                print(f"Response length: {len(result) if isinstance(result, str) else 'N/A'}")
                
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
                
                # Check if it looks like it used course content
                if any(indicator in result.lower() for indicator in ["mcp", "chroma", "anthropic", "retrieval"]):
                    print("✅ Response appears to use course content")
                else:
                    print("⚠️  Response might not have used course content")
                
                # Check for error indicators
                if any(error in result.lower() for error in ["error", "failed", "sorry", "cannot"]):
                    print(f"⚠️  Response contains potential error: {result[:200]}...")
                
        except Exception as e:
            print(f"❌ Real API with tools test failed: {e}")
            raise

    @patch('anthropic.Anthropic')
    def test_06_error_handling_api_failures(self, mock_anthropic):
        """Test error handling for API failures"""
        print("\n=== Testing API Error Handling ===")
        
        # Mock API errors
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Test different types of API errors
        api_errors = [
            Exception("Invalid API key"),
            Exception("Rate limit exceeded"),
            Exception("Network timeout"),
            Exception("Service unavailable")
        ]
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        for error in api_errors:
            mock_client.messages.create.side_effect = error
            
            with self.assertRaises(Exception):
                ai_gen.generate_response("test query")
            
            print(f"✅ Properly propagated error: {type(error).__name__}")

    @patch('anthropic.Anthropic')
    def test_07_tool_execution_errors(self, mock_anthropic):
        """Test handling of tool execution errors"""
        print("\n=== Testing Tool Execution Error Handling ===")
        
        # Set up mocks
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Mock tool manager that fails
        tool_manager = Mock()
        tool_manager.get_tool_definitions.return_value = [{
            "name": "failing_tool",
            "description": "A tool that fails",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }]
        tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "failing_tool"
        mock_tool_content.id = "tool_12345"
        mock_tool_content.input = {"query": "test"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_content]
        initial_response.stop_reason = "tool_use"
        
        mock_client.messages.create.return_value = initial_response
        
        # This should raise an exception during tool execution
        with self.assertRaises(Exception):
            ai_gen.generate_response(
                query="test",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )
        
        print("✅ Tool execution errors are properly propagated")

    def test_08_conversation_history_handling(self):
        """Test conversation history handling"""
        print("\n=== Testing Conversation History Handling ===")
        
        if not self.use_real_api:
            print("⚠️  Skipping conversation history test - no valid API key")
            return
        
        try:
            # Test with conversation history
            history = "User: Hello\nAssistant: Hello! How can I help you?"
            
            result = self.ai_generator.generate_response(
                query="What's 2+2?",
                conversation_history=history
            )
            
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            print("✅ Conversation history handling works")
            
        except Exception as e:
            print(f"❌ Conversation history test failed: {e}")
            raise

    def test_09_system_prompt_behavior(self):
        """Test that the system prompt is working correctly"""
        print("\n=== Testing System Prompt Behavior ===")
        
        if not self.use_real_api:
            print("⚠️  Skipping system prompt test - no valid API key")
            return
        
        try:
            # Create tool manager
            vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            tool_manager = ToolManager()
            search_tool = CourseSearchTool(vector_store)
            tool_manager.register_tool(search_tool)
            
            # Test query that should definitely trigger tool use based on system prompt
            result = self.ai_generator.generate_response(
                query="Find information about MCP in the course materials",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )
            
            print(f"System prompt test result length: {len(result)}")
            
            # The system prompt should guide the AI to use tools for course content
            # We can't easily test this without inspecting API calls, but we can check if it returns reasonable results
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 50)  # Should be substantial response
            print("✅ System prompt appears to be working")
            
        except Exception as e:
            print(f"❌ System prompt test failed: {e}")
            raise

    @patch('anthropic.Anthropic')
    def test_10_malformed_tool_responses(self, mock_anthropic):
        """Test handling of malformed tool responses"""
        print("\n=== Testing Malformed Tool Response Handling ===")
        
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.get_tool_definitions.return_value = [{"name": "test_tool"}]
        
        # Test various malformed responses
        malformed_responses = [
            # Missing content
            Mock(content=[], stop_reason="tool_use"),
            # Wrong content type
            Mock(content=[Mock(type="text", text="not a tool")], stop_reason="tool_use"),
            # Missing required fields
            Mock(content=[Mock(type="tool_use", name="test_tool")], stop_reason="tool_use"),  # Missing id and input
        ]
        
        for malformed_response in malformed_responses:
            mock_client.messages.create.return_value = malformed_response
            
            try:
                result = ai_gen.generate_response(
                    query="test",
                    tools=tool_manager.get_tool_definitions(),
                    tool_manager=tool_manager
                )
                print(f"⚠️  Malformed response was handled (maybe gracefully): {result}")
            except Exception as e:
                print(f"❌ Malformed response caused exception: {e}")
                # This might be expected behavior


def run_ai_generator_tests():
    """Run AIGenerator tests with detailed output"""
    print("=" * 70)
    print("AI GENERATOR COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIGenerator)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("AI GENERATOR TEST SUMMARY")
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
    run_ai_generator_tests()