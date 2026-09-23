# Azure Hybrid RAG System — Viva Questions and Answers

## 1. What is RAG?
RAG stands for Retrieval-Augmented Generation. It retrieves relevant information and supplies it to a language model before generating an answer.

## 2. Why use RAG?
RAG allows a language model to use information from a project-specific document collection during question answering.

## 3. Why is OCR needed?
Scanned PDFs may contain page images instead of selectable text. OCR converts visible text into machine-readable text.

## 4. Why use PyMuPDF?
PyMuPDF provides native PDF text extraction and page-level processing.

## 5. Why use embeddings?
Embeddings represent text numerically so semantically similar questions and document chunks can be compared.

## 6. Why use FAISS?
FAISS provides efficient similarity search over vector embeddings.

## 7. What is the purpose of chunking?
Chunking divides long documents into smaller pieces so relevant sections can be retrieved and supplied to the language model.

## 8. What are the three reasoning modes?
The system provides Documents Only, Hybrid, and General Knowledge modes.

## 9. What is Documents Only mode?
It instructs the model to answer using the supplied document context and avoid unsupported outside knowledge.

## 10. What is Hybrid mode?
It treats uploaded documents as the primary source while allowing general knowledge when the retrieved context is insufficient.

## 11. What is General Knowledge mode?
It allows the model to answer using general knowledge without treating uploaded documents as evidence.

## 12. Does the system completely eliminate hallucinations?
No. Prompt grounding, retrieval, source attribution, and mode-specific instructions are used to reduce unsupported responses, but no generative system can be guaranteed to be completely hallucination-free.

## 13. Why use Azure OpenAI?
Azure OpenAI provides the project's language-model and embedding capabilities through Azure.

## 14. Why use Streamlit?
Streamlit provides a practical interactive interface for the complete Python-based RAG workflow.

## 15. What happens when a scanned PDF is uploaded?
The system first attempts native PDF text extraction. When usable text is not available, OCR is used as a fallback.

## 16. What information is retained with chunks?
The ingestion pipeline keeps metadata such as the source filename and page number so retrieved information can be attributed to its source.

## 17. What is the complete pipeline?
PDF upload → extraction/OCR → chunking → embeddings → FAISS retrieval → prompt/context construction → Azure OpenAI → final answer with sources.
