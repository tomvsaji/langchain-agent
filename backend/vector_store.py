"""Vector database module using ChromaDB for document storage and retrieval."""

from __future__ import annotations

from typing import Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K


class VectorStore:
    """Manages a ChromaDB collection for storing and querying documents."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._persist_dir = persist_directory or CHROMA_PERSIST_DIR
        self._collection_name = collection_name or CHROMA_COLLECTION
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Split texts into chunks and add them to the vector store.

        Returns the list of chunk IDs that were inserted.
        """
        all_chunks: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str] = []

        for idx, text in enumerate(texts):
            chunks = self._splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = (metadatas[idx] if metadatas else {}).copy()
                meta["source_index"] = idx
                meta["chunk_index"] = chunk_idx
                all_metas.append(meta)
                if ids:
                    all_ids.append(f"{ids[idx]}_chunk{chunk_idx}")
                else:
                    all_ids.append(f"doc{idx}_chunk{chunk_idx}")

        if all_chunks:
            self._collection.add(
                documents=all_chunks,
                metadatas=all_metas,
                ids=all_ids,
            )
        return all_ids

    def similarity_search(
        self, query: str, top_k: Optional[int] = None
    ) -> list[dict]:
        """Return the top-k most similar documents to *query*.

        Each result dict contains ``document``, ``metadata``, ``distance``.
        """
        k = top_k or RETRIEVAL_TOP_K
        count = self._collection.count()
        if count == 0:
            return []

        # Don't request more results than documents in the collection
        k = min(k, count)

        results = self._collection.query(query_texts=[query], n_results=k)
        hits: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"document": doc, "metadata": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
        )

    def seed_sample_documents(self) -> list[str]:
        """Insert a small set of sample documents for demo purposes."""
        sample_docs = [
            (
                "LangChain is a framework for developing applications powered by "
                "language models. It provides tools for prompt management, chains, "
                "agents, memory, and retrieval-augmented generation."
            ),
            (
                "ChromaDB is an open-source vector database designed for AI "
                "applications. It stores embeddings alongside metadata and "
                "supports fast similarity search."
            ),
            (
                "Retrieval-Augmented Generation (RAG) combines a retrieval step "
                "with a generation step. Relevant documents are fetched from a "
                "knowledge base and provided as context to the language model."
            ),
            (
                "FastAPI is a modern, high-performance web framework for building "
                "APIs with Python. It uses type hints for automatic validation "
                "and OpenAPI documentation generation."
            ),
            (
                "Streamlit is an open-source Python library that makes it easy "
                "to create and share custom web apps for machine learning and "
                "data science projects."
            ),
        ]
        ids = [f"sample_{i}" for i in range(len(sample_docs))]
        metadatas = [{"source": "seed_data"} for _ in sample_docs]
        return self.add_documents(sample_docs, metadatas=metadatas, ids=ids)
