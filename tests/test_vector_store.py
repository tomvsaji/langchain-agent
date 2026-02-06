"""Tests for the ChromaDB vector store module."""

from backend.vector_store import VectorStore


class TestAddDocuments:
    def test_add_single_short_document(self, vector_store: VectorStore):
        ids = vector_store.add_documents(["Short document."])
        assert len(ids) == 1
        assert vector_store.count() == 1

    def test_add_multiple_documents(self, vector_store: VectorStore):
        ids = vector_store.add_documents(["Doc A", "Doc B", "Doc C"])
        assert len(ids) == 3
        assert vector_store.count() == 3

    def test_add_with_metadata(self, vector_store: VectorStore):
        ids = vector_store.add_documents(
            ["With meta"], metadatas=[{"source": "test"}]
        )
        assert len(ids) == 1

    def test_add_with_custom_ids(self, vector_store: VectorStore):
        ids = vector_store.add_documents(["Custom ID"], ids=["myid"])
        assert ids[0].startswith("myid_chunk")

    def test_long_document_is_chunked(self, vector_store: VectorStore):
        long_text = "word " * 500  # ~2500 chars, should be split into chunks
        ids = vector_store.add_documents([long_text])
        assert len(ids) > 1  # Should produce multiple chunks
        assert vector_store.count() > 1


class TestSimilaritySearch:
    def test_search_empty_store(self, vector_store: VectorStore):
        results = vector_store.similarity_search("anything")
        assert results == []

    def test_search_returns_results(self, vector_store: VectorStore):
        vector_store.add_documents(
            ["Python is a programming language", "Java is also a language"]
        )
        results = vector_store.similarity_search("Python programming")
        assert len(results) >= 1
        assert "document" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_search_respects_top_k(self, vector_store: VectorStore):
        vector_store.add_documents(["A", "B", "C", "D", "E"])
        results = vector_store.similarity_search("test", top_k=2)
        assert len(results) == 2

    def test_search_top_k_exceeds_count(self, vector_store: VectorStore):
        vector_store.add_documents(["Only one doc"])
        results = vector_store.similarity_search("query", top_k=10)
        assert len(results) == 1


class TestResetAndSeed:
    def test_reset_clears_all(self, vector_store: VectorStore):
        vector_store.add_documents(["A", "B"])
        assert vector_store.count() == 2
        vector_store.reset()
        assert vector_store.count() == 0

    def test_seed_sample_documents(self, vector_store: VectorStore):
        ids = vector_store.seed_sample_documents()
        assert len(ids) >= 5  # At least 5 sample docs (may have chunks)
        assert vector_store.count() >= 5

    def test_seed_then_search(self, vector_store: VectorStore):
        vector_store.seed_sample_documents()
        results = vector_store.similarity_search("vector database")
        assert len(results) >= 1
        # The ChromaDB doc should be among top results
        combined = " ".join(r["document"] for r in results)
        assert "chroma" in combined.lower() or "vector" in combined.lower()
