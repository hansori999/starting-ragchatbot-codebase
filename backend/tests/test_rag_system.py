import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


class TestMaxResultsBug:
    """Tests documenting the MAX_RESULTS=0 bug."""

    def test_max_results_zero_guarded_by_fallback(self):
        """Defensive guard: VectorStore with max_results=0 falls back to 5 instead of failing."""
        mock_chroma_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["chunk 1"]],
            "metadatas": [[{"course_title": "Test", "lesson_number": 1}]],
            "distances": [[0.1]]
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("vector_store.chromadb.PersistentClient", return_value=mock_chroma_client), \
             patch("vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            from vector_store import VectorStore
            store = VectorStore(
                chroma_path="/tmp/test_chroma",
                embedding_model="test-model",
                max_results=0
            )

            results = store.search(query="test query")

            assert results.error is None
            # Verify the fallback sent n_results=5 to ChromaDB
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args.kwargs
            assert call_kwargs["n_results"] == 5

    def test_max_results_positive_searches_succeed(self):
        """VectorStore with max_results=5 succeeds when ChromaDB returns results."""
        mock_chroma_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["chunk 1", "chunk 2"]],
            "metadatas": [[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2},
            ]],
            "distances": [[0.1, 0.2]]
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("vector_store.chromadb.PersistentClient", return_value=mock_chroma_client), \
             patch("vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            from vector_store import VectorStore
            store = VectorStore(
                chroma_path="/tmp/test_chroma",
                embedding_model="test-model",
                max_results=5
            )

            results = store.search(query="test query")

            assert results.error is None
            assert len(results.documents) == 2

    def test_config_max_results_is_positive(self):
        """Verifies config.MAX_RESULTS is a positive value (bug was MAX_RESULTS=0)."""
        from config import config
        assert config.MAX_RESULTS > 0, (
            f"MAX_RESULTS must be positive, got {config.MAX_RESULTS}"
        )


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() orchestration."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_RESULTS = 5
        config.CHROMA_PATH = "/tmp/test"
        config.EMBEDDING_MODEL = "test-model"
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "test-model"
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def rag_system(self, mock_config):
        """RAGSystem with all components mocked."""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore") as MockVS, \
             patch("rag_system.AIGenerator") as MockAI, \
             patch("rag_system.SessionManager") as MockSM:

            mock_vs_instance = MockVS.return_value
            mock_ai_instance = MockAI.return_value
            mock_sm_instance = MockSM.return_value

            mock_ai_instance.generate_response.return_value = "AI response"
            mock_sm_instance.get_conversation_history.return_value = None

            from rag_system import RAGSystem
            system = RAGSystem(mock_config)

            # Expose mocks for assertions
            system._mock_ai = mock_ai_instance
            system._mock_sm = mock_sm_instance
            system._mock_vs = mock_vs_instance
            return system

    def test_query_calls_ai_with_tools(self, rag_system):
        """RAGSystem.query() passes tool definitions and tool_manager to ai_generator."""
        rag_system.query("What is AI?")

        call_kwargs = rag_system._mock_ai.generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is rag_system.tool_manager

    def test_query_extracts_and_resets_sources(self, rag_system):
        """After query, get_last_sources() called then reset_sources() called."""
        # Spy on tool_manager methods
        rag_system.tool_manager.get_last_sources = MagicMock(return_value=[{"text": "Source", "link": "url"}])
        rag_system.tool_manager.reset_sources = MagicMock()

        response, sources = rag_system.query("What is AI?")

        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()
        assert sources == [{"text": "Source", "link": "url"}]

    def test_query_updates_session_history(self, rag_system):
        """session_manager.add_exchange() called with query and response."""
        rag_system.query("What is AI?", session_id="session_1")

        rag_system._mock_sm.add_exchange.assert_called_once()
        call_args = rag_system._mock_sm.add_exchange.call_args
        assert call_args[0][0] == "session_1"  # session_id
        assert "What is AI?" in call_args[0][1]  # query (wrapped in prompt)
        assert call_args[0][2] == "AI response"  # response

    def test_query_without_session(self, rag_system):
        """Works when session_id=None, no session methods called."""
        response, sources = rag_system.query("What is AI?")

        assert response == "AI response"
        rag_system._mock_sm.get_conversation_history.assert_not_called()
        rag_system._mock_sm.add_exchange.assert_not_called()
