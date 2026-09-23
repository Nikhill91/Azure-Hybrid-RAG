# Azure Hybrid RAG System — Project Documentation

## 1. Project Overview

Azure Hybrid RAG System is a Retrieval-Augmented Generation application designed to answer questions using uploaded PDF documents while also supporting general-knowledge queries.

The system combines document ingestion, OCR, chunking, embeddings, FAISS vector retrieval, Azure OpenAI, and a Streamlit interface.

## 2. Problem Statement

Conventional RAG systems can face difficulties with:
- Scanned PDFs where normal text extraction does not work.
- Questions that require information beyond the uploaded documents.
- Keeping retrieved evidence connected to the final answer.
- Providing useful source attribution.

## 3. Solution

The project uses a hybrid pipeline:

PDF upload → PyMuPDF extraction → Tesseract OCR fallback → chunking → Azure embeddings → FAISS retrieval → Azure OpenAI → answer with source attribution.

## 4. Main Components

### Document Ingestion
PyMuPDF is used for native PDF text extraction. Tesseract OCR is used as a fallback for scanned pages. Extracted chunks retain document and page metadata.

### Embeddings and Vector Search
Document chunks are converted into embeddings using Azure OpenAI's `text-embedding-3-large` model. FAISS is used for vector similarity search.

### LLM and Reasoning
Azure OpenAI GPT-5-mini generates responses using the selected reasoning mode.

### Streamlit Application
Streamlit provides the user interface for uploading documents, selecting a mode, asking questions, viewing answers, and viewing sources.

## 5. Reasoning Modes

### Documents Only
The model is instructed to answer strictly from the supplied document context and avoid unsupported outside knowledge.

### Hybrid
Uploaded documents are treated as the primary source. General knowledge can be used when the retrieved documents are insufficient, and the distinction should be made clear.

### General Knowledge
The model answers using general knowledge and does not treat uploaded documents as evidence.

## 6. Technology Stack

- Python
- Streamlit
- Azure OpenAI
- GPT-5-mini
- text-embedding-3-large
- FAISS
- PyMuPDF
- Tesseract OCR

## 7. Project Workflow

1. User uploads PDF files.
2. PDF pages are processed.
3. Native text is extracted where available.
4. OCR is used for scanned pages when required.
5. Text is divided into chunks.
6. Chunks are converted into embeddings.
7. Embeddings are stored and searched with FAISS.
8. A user question is embedded and relevant chunks are retrieved.
9. The selected reasoning mode determines how context is used.
10. Azure OpenAI generates the final response.
11. Source information is displayed with the answer.

## 8. Team Contributions

- **Nikhil:** PDF ingestion, OCR, chunking, and team leadership.
- **Sameer:** Embeddings, vector search, and knowledge-base layer.
- **Harsh:** LLM integration, reasoning modes, and prompt grounding.
- **Aryan:** Streamlit UI and system integration.
- **Bhuvnesh:** Project coordination, documentation, demonstration planning, and viva preparation.

## 9. Expected Outcome

The completed application provides an interactive RAG workflow that can process PDF documents, retrieve relevant information, generate answers through Azure OpenAI, and expose source information to the user.
