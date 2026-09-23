# Azure Hybrid RAG System — 12-Slide Presentation Outline

## Slide 1 — Title
Azure Hybrid RAG System

Team:
Nikhil • Sameer • Harsh • Aryan • Bhuvnesh

## Slide 2 — Problem Statement
- PDF question answering can fail when documents are scanned.
- Traditional RAG can be limited to retrieved document context.
- Users need clearer control over document-grounded and general-knowledge responses.

## Slide 3 — Project Objective
- Build an interactive PDF-based RAG system.
- Support scanned and native PDFs.
- Retrieve semantically relevant content.
- Provide three reasoning modes.
- Display source attribution.

## Slide 4 — High-Level Architecture
PDF Upload
→ PyMuPDF
→ OCR fallback
→ Chunking
→ Azure Embeddings
→ FAISS
→ Retrieval
→ Prompt/Context
→ Azure OpenAI
→ Streamlit Answer

## Slide 5 — Document Ingestion
- PyMuPDF for native extraction.
- Tesseract OCR for scanned content.
- Page metadata retained.
- Text divided into chunks.

## Slide 6 — Embeddings & Vector Search
- text-embedding-3-large.
- Vector representation of chunks.
- FAISS similarity search.
- Relevant chunks retrieved for each query.

## Slide 7 — Three Reasoning Modes
Documents Only:
- Use supplied document context.

Hybrid:
- Documents are primary.
- General knowledge can supplement when needed.

General Knowledge:
- Answer using general knowledge.

## Slide 8 — Prompt Grounding
- Mode-specific instructions.
- Avoid unsupported document claims.
- Avoid invented citations.
- Use source information supplied by retrieval.

## Slide 9 — Streamlit Application
Show:
- Upload area.
- Processing status.
- Mode selector.
- Chat interface.
- Source cards.
- Error/status handling.

## Slide 10 — Demonstration
Demonstrate:
1. Upload PDF.
2. Documents Only query.
3. Hybrid query.
4. General Knowledge query.
5. Source attribution.

## Slide 11 — Team Contributions
Nikhil — ingestion/OCR/chunking
Sameer — embeddings/vector search
Harsh — LLM/prompts/modes
Aryan — UI/integration
Bhuvnesh — documentation/demo/coordination

## Slide 12 — Conclusion
- Complete end-to-end RAG workflow.
- Handles native and scanned PDFs.
- Provides multiple reasoning modes.
- Combines retrieval with Azure OpenAI.
- Provides source attribution.

Thank You
