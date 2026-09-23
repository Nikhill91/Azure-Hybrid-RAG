# Azure Hybrid RAG System — Demo Guide

## Target Duration

Maximum: 5 minutes.

## Demo Sequence

### 0:00–0:30 — Introduction
Show the project title and briefly explain the problem:
traditional document QA can struggle with scanned PDFs and questions that go beyond retrieved documents.

### 0:30–1:15 — Upload and Processing
Open the Streamlit application.

Show:
- PDF upload
- Processing
- Knowledge-base status

Briefly explain that PyMuPDF handles native extraction and Tesseract OCR provides a fallback for scanned content.

### 1:15–2:15 — Documents Only
Select **Documents Only**.

Ask a question whose answer is clearly present in the uploaded PDF.

Point out:
- The generated answer.
- The document/page source information.

Explain that this mode is intended to keep the answer grounded in supplied document context.

### 2:15–3:15 — Hybrid Mode
Switch to **Hybrid**.

Ask a question where the uploaded document provides useful context but additional general knowledge is helpful.

Explain that documents are treated as the primary source while general knowledge can be used when the retrieved context is insufficient.

### 3:15–4:00 — General Knowledge
Switch to **General Knowledge**.

Ask a question that does not depend on the uploaded PDF.

Explain that this mode is intended for general-knowledge responses rather than treating uploaded documents as evidence.

### 4:00–4:30 — Source Attribution and UI
Show the source cards and the rest of the Streamlit interface.

Mention that the interface helps users see where retrieved information came from.

### 4:30–5:00 — Conclusion
Summarize:

PDF → extraction/OCR → chunks → embeddings → FAISS → retrieval → Azure OpenAI → answer + sources.

End with the project title and team names.

## Demo Tips

- Use a PDF that the team has already tested.
- Prepare the exact questions before recording.
- Avoid spending time waiting for long processing steps.
- Keep the video below the required five-minute maximum.
