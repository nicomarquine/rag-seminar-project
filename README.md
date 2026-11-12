# Retrieval-Augmented Generation (RAG) Benchmark Pipeline

A containerized Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**, **ChromaDB**, and **Google Gemini API**, orchestrated via **Docker Compose**.

This project was developed as the practical component for the Cloud Computing individual seminar paper at Aspira University College. It implements an automated evaluation benchmark that measures **Retrieval Accuracy (Hit Rate)** and **end-to-end latency** across different text chunking and retrieval configurations.

---

## Architecture Overview

The system deploys two isolated services inside a shared Docker network:
1. **`chromadb`**: An open-source vector database running as an HTTP server (`port 8000`) with persistent local volume storage.
2. **`rag-app`**: A Python application that loads domain-specific documents, generates dense vector embeddings (`text-embedding-004`), queries ChromaDB, and generates grounded responses using `gemini-2.5-flash`.

```text
[ sample_data.txt ] ──( Chunking )──> [ Embeddings ] ──> [ ChromaDB Container ]
                                                                 │
[ Ground Truth Q&A ] ────────────────────────────────────> [ Retriever (Top-k) ]
                                                                 │
                                                         [ Gemini 2.5 Flash ]
                                                                 │
                                                   [ Retrieval Accuracy & Latency ]