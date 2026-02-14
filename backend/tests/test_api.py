"""Tests for FastAPI API endpoints.

Defines a standalone test app that mirrors the routes in app.py but avoids
importing app.py directly (which mounts static files from a path that doesn't
exist in the test environment).
"""

import sys
import os
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Inline test app — mirrors backend/app.py routes without static file mount
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system):
    """Build a minimal FastAPI app wired to the given mock RAG system."""
    app = FastAPI()

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(mock_rag_system):
    """TestClient backed by a mock RAG system."""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/query endpoint
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Tests for POST /api/query."""

    def test_query_returns_answer_and_sources(self, client):
        """Successful query returns 200 with answer, sources, and session_id."""
        resp = client.post("/api/query", json={"query": "What is AI?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "This is a test answer."
        assert data["sources"] == ["Intro to AI - Lesson 1"]
        assert data["session_id"] == "session_42"

    def test_query_with_session_id(self, client, mock_rag_system):
        """When session_id is provided, it is passed through and returned."""
        resp = client.post("/api/query", json={"query": "Follow-up", "session_id": "my_session"})

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my_session"
        mock_rag_system.query.assert_called_once_with("Follow-up", "my_session")

    def test_query_creates_session_when_missing(self, client, mock_rag_system):
        """When no session_id is sent, a new session is created."""
        resp = client.post("/api/query", json={"query": "Hello"})

        assert resp.status_code == 200
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_missing_body_returns_422(self, client):
        """Request with no JSON body returns 422 Unprocessable Entity."""
        resp = client.post("/api/query")
        assert resp.status_code == 422

    def test_query_missing_query_field_returns_422(self, client):
        """Request missing required 'query' field returns 422."""
        resp = client.post("/api/query", json={"session_id": "abc"})
        assert resp.status_code == 422

    def test_query_rag_error_returns_500(self, client, mock_rag_system):
        """When rag_system.query() raises, endpoint returns 500."""
        mock_rag_system.query.side_effect = RuntimeError("Model API unavailable")

        resp = client.post("/api/query", json={"query": "Will fail"})

        assert resp.status_code == 500
        assert "Model API unavailable" in resp.json()["detail"]

    def test_query_empty_string(self, client, mock_rag_system):
        """Empty-string query is still forwarded to RAG system (validation is app-level)."""
        resp = client.post("/api/query", json={"query": ""})

        assert resp.status_code == 200
        mock_rag_system.query.assert_called_once()


# ---------------------------------------------------------------------------
# /api/courses endpoint
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    """Tests for GET /api/courses."""

    def test_courses_returns_stats(self, client):
        """Successful request returns course count and titles."""
        resp = client.get("/api/courses")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 3
        assert "Intro to AI" in data["course_titles"]
        assert len(data["course_titles"]) == 3

    def test_courses_error_returns_500(self, client, mock_rag_system):
        """When get_course_analytics() raises, endpoint returns 500."""
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB connection lost")

        resp = client.get("/api/courses")

        assert resp.status_code == 500
        assert "DB connection lost" in resp.json()["detail"]

    def test_courses_empty_catalog(self, client, mock_rag_system):
        """Empty course catalog returns zero counts."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        resp = client.get("/api/courses")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


# ---------------------------------------------------------------------------
# Unknown routes
# ---------------------------------------------------------------------------

class TestUnknownRoutes:
    """Tests for routes not defined in the API."""

    def test_unknown_route_returns_404(self, client):
        """GET to undefined path returns 404."""
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_wrong_method_returns_405(self, client):
        """GET on POST-only endpoint returns 405."""
        resp = client.get("/api/query")
        assert resp.status_code == 405
