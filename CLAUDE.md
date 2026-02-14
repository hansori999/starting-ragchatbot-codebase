# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack RAG (Retrieval-Augmented Generation) chatbot that lets users query course materials using semantic search and Anthropic's Claude API. Python/FastAPI backend with a vanilla HTML/CSS/JS frontend.

## Development Commands

```bash
# Install dependencies
uv sync

# Run the server (from project root)
./run.sh
# Or manually from backend/:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points
# Web UI: http://localhost:8000
# API docs: http://localhost:8000/docs
```

No test suite, linter, or build step is configured.

## Architecture

**Request flow:** Frontend → `POST /api/query` → `RAGSystem.query()` → VectorStore semantic search → Claude API with tool use → response with sources

### Backend (`backend/`)

| File | Role |
|------|------|
| `app.py` | FastAPI app, routes (`/api/query`, `/api/courses`), serves frontend static files, loads docs on startup |
| `rag_system.py` | Main orchestrator — initializes all components, deduplicates courses, delegates to vector store and AI generator |
| `document_processor.py` | Parses structured course text files from `docs/`, chunks by sentences (800 chars, 100 overlap) |
| `vector_store.py` | ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunks). Supports filtered search by course name and lesson number |
| `ai_generator.py` | Claude API client — handles the tool-use loop (initial call → tool execution → final response). Temperature 0, max 800 tokens |
| `search_tools.py` | Tool abstraction layer. `CourseSearchTool` wraps vector search for Claude's tool use. `ToolManager` registers and dispatches tools |
| `session_manager.py` | In-memory conversation history (max 5 exchanges per session, auto-cleanup) |
| `config.py` | Central config loaded from `.env`. Key settings: chunk size, model names, ChromaDB path |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |

### Frontend (`frontend/`)

Vanilla HTML/CSS/JS. Dark-themed UI with sidebar (course stats, suggested questions) and chat area. Uses Marked.js for markdown rendering. Communicates via fetch to backend API.

### Data (`docs/`)

Structured text files with a specific format: course metadata headers (`Course Title:`, `Course Link:`, `Course Instructor:`) followed by lesson sections (`Lesson N: Title`). The document processor depends on this format.

## Key Design Decisions

- **Tool use pattern:** Claude receives a `search_course_content` tool and decides when to search. System prompt enforces single search per query.
- **Vector store:** ChromaDB persists to `backend/chroma_db/` (gitignored). Uses `all-MiniLM-L6-v2` sentence transformer for embeddings.
- **Session state is in-memory only** — lost on server restart.
- **Static file serving:** FastAPI serves the frontend directly (no separate dev server).

## Environment Setup

- Python 3.13+ required (see `.python-version`)
- `uv` package manager required
- Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`
- **Always use `uv` — never use `pip` directly.** Use `uv run` to execute Python files and `uv add` to install dependencies.
