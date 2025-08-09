"""
Diagnostic tests to verify document loading and ChromaDB state.
This test helps identify what's currently broken in the system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore
from test_helpers import check_chroma_db_state, test_actual_api_connectivity


class TestDataLoadingAndState(unittest.TestCase):
    """Test data loading and current system state"""

    def setUp(self):
        """Set up test environment"""
        self.docs_path = "../docs"
        self.chroma_path = config.CHROMA_PATH
        self.api_key = config.ANTHROPIC_API_KEY

    def test_01_docs_folder_exists_and_has_content(self):
        """Test that docs folder exists and contains expected files"""
        print(f"\n=== Testing docs folder: {self.docs_path} ===")
        
        # Check if docs folder exists
        docs_full_path = os.path.join(os.path.dirname(__file__), self.docs_path)
        self.assertTrue(os.path.exists(docs_full_path), 
                       f"Docs folder not found at {docs_full_path}")
        
        # List contents
        files = os.listdir(docs_full_path)
        print(f"Files in docs folder: {files}")
        
        # Check for expected course files
        course_files = [f for f in files if f.endswith(('.txt', '.pdf', '.docx'))]
        self.assertGreater(len(course_files), 0, 
                          f"No course files found in {docs_full_path}")
        
        # Check file sizes to ensure they're not empty
        for filename in course_files:
            file_path = os.path.join(docs_full_path, filename)
            file_size = os.path.getsize(file_path)
            print(f"  {filename}: {file_size} bytes")
            self.assertGreater(file_size, 0, f"File {filename} is empty")

    def test_02_chroma_db_state_check(self):
        """Check current state of ChromaDB"""
        print(f"\n=== Checking ChromaDB state: {self.chroma_path} ===")
        
        chroma_state = check_chroma_db_state(self.chroma_path)
        print(f"ChromaDB State: {chroma_state}")
        
        # Check if ChromaDB is accessible
        if not chroma_state["chroma_accessible"]:
            print(f"⚠️  ChromaDB not accessible: {chroma_state.get('error')}")
            return
        
        # Check collections
        collections = chroma_state.get("collections", {})
        
        # Course catalog check
        course_catalog_info = collections.get("course_catalog", {})
        if course_catalog_info.get("exists"):
            count = course_catalog_info.get("count", 0)
            print(f"✅ Course catalog exists with {count} entries")
            if count > 0:
                sample_ids = course_catalog_info.get("sample_ids", [])
                print(f"   Sample course IDs: {sample_ids}")
            else:
                print("⚠️  Course catalog is empty")
        else:
            print(f"❌ Course catalog missing: {course_catalog_info.get('error')}")
        
        # Course content check
        course_content_info = collections.get("course_content", {})
        if course_content_info.get("exists"):
            count = course_content_info.get("count", 0)
            print(f"✅ Course content exists with {count} chunks")
            if count > 0:
                sample_ids = course_content_info.get("sample_ids", [])
                print(f"   Sample chunk IDs: {sample_ids}")
            else:
                print("⚠️  Course content is empty")
        else:
            print(f"❌ Course content missing: {course_content_info.get('error')}")

    def test_03_api_key_and_connectivity(self):
        """Test API key configuration and connectivity"""
        print(f"\n=== Testing API Configuration ===")
        
        # Check API key exists
        print(f"API Key configured: {'Yes' if self.api_key else 'No'}")
        if not self.api_key:
            print("❌ ANTHROPIC_API_KEY is missing or empty")
            self.skipTest("API key not configured")
            return
        
        # Mask the API key for security
        masked_key = self.api_key[:8] + "..." + self.api_key[-8:] if len(self.api_key) > 16 else "***"
        print(f"API Key: {masked_key}")
        
        # Test actual API connectivity (only if key looks valid)
        if len(self.api_key) > 20:  # Basic validation
            print("Testing API connectivity...")
            api_result = test_actual_api_connectivity(self.api_key)
            print(f"API Test Result: {api_result}")
            
            if api_result.get("api_accessible"):
                print("✅ API is accessible")
            else:
                error = api_result.get("error", "Unknown error")
                print(f"❌ API connection failed: {error}")

    def test_04_rag_system_initialization(self):
        """Test RAG system initialization"""
        print(f"\n=== Testing RAG System Initialization ===")
        
        try:
            rag = RAGSystem(config)
            print("✅ RAG System initialized successfully")
            
            # Test tool registration
            tool_definitions = rag.tool_manager.get_tool_definitions()
            print(f"Registered tools: {len(tool_definitions)}")
            for tool_def in tool_definitions:
                tool_name = tool_def.get("name", "unknown")
                print(f"  - {tool_name}")
            
            # Test course analytics
            analytics = rag.get_course_analytics()
            print(f"Course analytics: {analytics}")
            
        except Exception as e:
            print(f"❌ RAG System initialization failed: {e}")
            raise

    def test_05_vector_store_search_functionality(self):
        """Test vector store search with simple queries"""
        print(f"\n=== Testing Vector Store Search ===")
        
        try:
            vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            
            # Test simple search
            test_queries = [
                "machine learning",
                "introduction",
                "lesson 1",
                "what is"
            ]
            
            for query in test_queries:
                print(f"\nTesting search: '{query}'")
                results = vector_store.search(query)
                
                if results.error:
                    print(f"  ❌ Search error: {results.error}")
                elif results.is_empty():
                    print(f"  ⚠️  No results found")
                else:
                    print(f"  ✅ Found {len(results.documents)} results")
                    for i, (doc, meta) in enumerate(zip(results.documents[:2], results.metadata[:2])):
                        course = meta.get("course_title", "unknown")
                        lesson = meta.get("lesson_number", "?")
                        snippet = doc[:100] + "..." if len(doc) > 100 else doc
                        print(f"    {i+1}. [{course} - Lesson {lesson}] {snippet}")
                        
        except Exception as e:
            print(f"❌ Vector store test failed: {e}")
            raise

    def test_06_course_search_tool_basic_functionality(self):
        """Test CourseSearchTool basic functionality"""
        print(f"\n=== Testing CourseSearchTool ===")
        
        try:
            from search_tools import CourseSearchTool
            vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
            search_tool = CourseSearchTool(vector_store)
            
            # Test tool definition
            tool_def = search_tool.get_tool_definition()
            print(f"Tool definition: {tool_def['name']}")
            
            # Test basic execution
            test_query = "what is machine learning"
            print(f"Testing tool execution with: '{test_query}'")
            
            result = search_tool.execute(test_query)
            print(f"Result type: {type(result)}")
            print(f"Result length: {len(result) if isinstance(result, str) else 'N/A'}")
            
            if isinstance(result, str):
                if "error" in result.lower() or "failed" in result.lower():
                    print(f"⚠️  Tool returned error: {result}")
                elif "no relevant content" in result.lower():
                    print(f"⚠️  Tool found no content: {result}")
                else:
                    print(f"✅ Tool executed successfully")
                    # Show first 200 characters of result
                    preview = result[:200] + "..." if len(result) > 200 else result
                    print(f"Preview: {preview}")
            
        except Exception as e:
            print(f"❌ CourseSearchTool test failed: {e}")
            raise

    def test_07_end_to_end_query_test(self):
        """Test complete end-to-end query flow"""
        print(f"\n=== Testing End-to-End Query Flow ===")
        
        # Skip if no API key
        if not self.api_key or len(self.api_key) < 20:
            print("⚠️  Skipping E2E test - API key not configured")
            return
        
        try:
            rag = RAGSystem(config)
            
            test_queries = [
                "What is machine learning?",
                "Tell me about lesson 1",
                "How does supervised learning work?"
            ]
            
            for query in test_queries:
                print(f"\nTesting query: '{query}'")
                try:
                    answer, sources = rag.query(query)
                    print(f"Answer type: {type(answer)}")
                    print(f"Answer length: {len(answer) if isinstance(answer, str) else 'N/A'}")
                    print(f"Sources count: {len(sources)}")
                    
                    if isinstance(answer, str):
                        if "query failed" in answer.lower():
                            print(f"❌ Query failed: {answer}")
                        else:
                            print(f"✅ Query succeeded")
                            preview = answer[:150] + "..." if len(answer) > 150 else answer
                            print(f"Preview: {preview}")
                    
                    if sources:
                        print(f"Sources: {sources}")
                    
                except Exception as e:
                    print(f"❌ Query failed with exception: {e}")
                    
        except Exception as e:
            print(f"❌ End-to-end test setup failed: {e}")
            raise


def run_diagnostic_tests():
    """Run diagnostic tests and print results"""
    print("=" * 60)
    print("RAG CHATBOT DIAGNOSTIC TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataLoadingAndState)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")
    
    return result


if __name__ == "__main__":
    run_diagnostic_tests()