# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start the application:**
```bash
./run.sh
```
This starts the FastAPI server on port 8000 with auto-reload enabled. Requires ANTHROPIC_API_KEY in `.env` file.

**Manual start (for development):**
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Access points:**
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** for course materials with a tool-based architecture where the AI can intelligently choose when to search vs. use general knowledge.

### Core Flow
1. **Frontend** (`frontend/`) - Simple HTML/CSS/JS chat interface
2. **FastAPI Backend** (`backend/app.py`) - RESTful API server with CORS
3. **RAG System** (`backend/rag_system.py`) - Main orchestrator
4. **AI Generator** (`backend/ai_generator.py`) - Claude API integration with tool calling
5. **Search Tools** (`backend/search_tools.py`) - Tool-based semantic search
6. **Vector Store** (`backend/vector_store.py`) - ChromaDB integration

### Key Architectural Patterns

**Tool-Based AI Architecture:**
- AI receives tool definitions and decides when to use them
- `CourseSearchTool` performs semantic search only when needed
- Two-stage Claude API calls: tool decision → tool execution → final response

**Session Management:**
- Conversation history maintained per session (`backend/session_manager.py`)
- Sessions created automatically, limited message history (configurable)

**Document Processing Pipeline:**
- `DocumentProcessor` chunks course materials with sentence-based splitting
- `Course`, `Lesson`, `CourseChunk` models structure the data
- ChromaDB stores both metadata and content chunks separately

**Configuration System:**
- Centralized config in `backend/config.py`
- Environment variable support via python-dotenv
- Configurable chunk sizes, embedding models, conversation history

## Key Integration Points

**Frontend-Backend Communication:**
- `/api/query` - POST endpoint for user queries, returns `{answer, sources, session_id}`
- `/api/courses` - GET endpoint for course analytics
- Session persistence maintained through `session_id`

**AI-Search Integration:**
- `ToolManager` registers available tools for Claude
- `search_course_content` tool performs vector similarity search
- Sources tracked automatically and returned to UI

**Vector Search Chain:**
- Query → ChromaDB semantic search → sentence-transformers embedding
- Results formatted with course/lesson context
- Supports course name and lesson number filtering

## Data Models

**Core Models** (`backend/models.py`):
- `Course` - Full course with lessons and metadata
- `Lesson` - Individual lesson with number and title
- `CourseChunk` - Text chunks for vector storage with course/lesson context

**Search Results** (`backend/vector_store.py`):
- `SearchResults` dataclass wraps ChromaDB responses
- Handles empty results and error states

## Environment Setup

Required `.env` file:
```
ANTHROPIC_API_KEY=your_api_key_here
```

**Data Location:**
- Course documents: `docs/` folder (TXT/PDF/DOCX supported)
- Vector database: `./chroma_db` (auto-created)
- Documents loaded automatically on server startup

## Frontend Architecture

**Single Page Application:**
- Pure HTML/CSS/JS (no framework)
- Markdown rendering for AI responses with `marked.js`
- Real-time course statistics display
- Session-based conversation continuity

**Key Features:**
- Auto-scrolling chat interface
- Collapsible sources display
- Loading states and error handling
- Suggested questions for user guidance
- always use uv to manage python packages, do not use pip directly
- don't run the server using ./run.sh I will start it myself