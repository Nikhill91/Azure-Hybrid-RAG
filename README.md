# Azure Hybrid RAG

Features:
- Multiple PDF upload
- Normal PDF text extraction
- OCR for scanned PDFs
- Azure text-embedding-3-large
- FAISS semantic retrieval
- Azure gpt-5-mini
- Hybrid mode: documents + general model knowledge
- Documents Only mode
- General Knowledge mode
- Source filename/page/similarity display

## Setup

Activate your environment:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

For scanned PDFs on macOS:

```bash
brew install tesseract
```

Verify:

```bash
tesseract --version
```

## Environment

Keep your real `.env` file private.

Required:

```text
AZURE_API_KEY=...
openai_endpoint=https://YOUR_RESOURCE.openai.azure.com/openai/v1
LLM_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-large
```

## Test

```bash
python test_azure.py
```

## Start

```bash
streamlit run streamlit_app.py
```

## Answer modes

### Hybrid
Uses relevant uploaded documents first, but can answer beyond them using general model knowledge.

### Documents Only
Strict RAG. Answers only from uploaded documents.

### General Knowledge
Does not use the uploaded document index.

## Important

This version rebuilds the FAISS index when Process Documents is clicked.
For a production system, incremental ingestion can be added later.
