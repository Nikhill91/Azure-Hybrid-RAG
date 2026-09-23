import os
import pickle
from typing import List, Dict, Tuple
import json
import time
import faiss
import numpy as np
import fitz
import pytesseract
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_ENDPOINT = os.getenv("openai_endpoint")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

DOCUMENTS_FOLDER = "documents"
VECTOR_STORE_FOLDER = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_FOLDER, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_STORE_FOLDER, "chunks.pkl")
MANIFEST_PATH = os.path.join(VECTOR_STORE_FOLDER, "manifest.json")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
SIMILARITY_THRESHOLD = 0.40
EMBED_BATCH_SIZE = 32

if not API_KEY:
    raise ValueError("AZURE_API_KEY is missing from .env")
if not OPENAI_ENDPOINT:
    raise ValueError("openai_endpoint is missing from .env")

client = OpenAI(api_key=API_KEY, base_url=OPENAI_ENDPOINT)


def extract_pdf_pages(pdf_path: str, filename: str) -> List[Dict]:
    """Extract normal PDF text; fall back to OCR for scanned pages."""
    document = fitz.open(pdf_path)
    pages = []

    for page_number in range(len(document)):
        page = document[page_number]
        text = " ".join((page.get_text("text") or "").split())

        if not text:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            text = " ".join(pytesseract.image_to_string(image).split())

        if text:
            pages.append({
                "source": filename,
                "page": page_number + 1,
                "text": text,
            })

    document.close()
    return pages


def load_all_pdfs() -> List[Dict]:
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    pdf_files = sorted(
        f for f in os.listdir(DOCUMENTS_FOLDER)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in documents/.")

    all_pages = []
    for filename in pdf_files:
        path = os.path.join(DOCUMENTS_FOLDER, filename)
        all_pages.extend(extract_pdf_pages(path, filename))

    if not all_pages:
        raise ValueError(
            "The PDFs contain no extractable text. "
            "OCR also returned no text."
        )

    return all_pages


def create_chunks(pages: List[Dict]) -> List[Dict]:
    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            chunk_text = text[start:start + CHUNK_SIZE].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                })

            if start + CHUNK_SIZE >= len(text):
                break

            start += step

    return chunks


def create_embeddings(texts: List[str]) -> np.ndarray:
    vectors = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        data = sorted(response.data, key=lambda x: x.index)
        vectors.extend(item.embedding for item in data)

    return np.asarray(vectors, dtype="float32")


def build_vector_store() -> Dict:
    pages = load_all_pdfs()
    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    embeddings = create_embeddings([c["text"] for c in chunks])
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return {
        "files": len({p["source"] for p in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
        "dimension": int(embeddings.shape[1]),
    }


def vector_store_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)


def load_vector_store() -> Tuple[faiss.Index, List[Dict]]:
    if not vector_store_exists():
        raise FileNotFoundError(
            "Knowledge base not found. Process documents first."
        )

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def retrieve(question: str, top_k: int = TOP_K) -> List[Dict]:
    index, chunks = load_vector_store()

    query_vector = create_embeddings([question])
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        score = float(score)
        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[int(idx)]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": score,
        })

    return results


def create_context(results: List[Dict]) -> str:
    return "\n\n---\n\n".join(
        f"SOURCE {i}\n"
        f"FILE: {r['source']}\n"
        f"PAGE: {r['page']}\n"
        f"CONTENT:\n{r['text']}"
        for i, r in enumerate(results, 1)
    )


def generate_answer(question: str, mode: str = "Hybrid") -> Dict:
    """
    Hybrid:
      - use relevant documents when available
      - use general model knowledge when needed
    Documents Only:
      - answer only from retrieved documents
    General Knowledge:
      - skip document retrieval
    """

    if mode == "General Knowledge":
        system_prompt = (
    "You are a helpful general-purpose AI assistant. "
    "Answer using your general knowledge and reasoning. "
    "Do not claim or imply that information came from uploaded documents. "
    "Do not invent sources or citations. "
    "If the question depends on current or time-sensitive information that you "
    "cannot verify, clearly state that limitation. "
    "If you are uncertain about a fact, say so rather than presenting a guess as certain."
)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return {
            "answer": response.choices[0].message.content or "",
            "sources": [],
            "mode": "general",
        }

    results = retrieve(question)

    if mode == "Documents Only":
        if not results:
            return {
                "answer": (
                    "I could not find enough relevant information "
                    "in the uploaded documents."
                ),
                "sources": [],
                "mode": "documents",
            }

        context = create_context(results)

        system_prompt = (
    "You are a strict document-grounded RAG assistant. "
    "Use ONLY the supplied document context to answer the question. "
    "Do not use outside knowledge to fill missing information. "
    "Do not guess, infer unsupported facts, or invent citations. "
    "Every factual claim based on the documents must be supported by the "
    "provided context. "
    "Cite supporting information using [filename.pdf, Page X]. "
    "If the documents do not contain enough information, respond exactly with "
    "\"I could not find enough relevant information in the uploaded documents.\" "
    "Do not force an answer when the evidence is insufficient."
)

        user_prompt = f"""
DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Answer using only the document context and include citations.
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return {
            "answer": response.choices[0].message.content or "",
            "sources": results,
            "mode": "documents",
        }

    # ---------------- HYBRID ----------------
    if results:
        context = create_context(results)

        system_prompt = (
    "You are a Hybrid RAG AI Assistant. "
    "Use relevant uploaded documents as the primary source whenever they contain "
    "information relevant to the question. "
    "You may use general knowledge when the retrieved documents are insufficient, "
    "but clearly distinguish general knowledge from information supported by the "
    "uploaded documents. "
    "Never invent document citations. "
    "Only cite a document when the supplied context actually supports the claim. "
    "Do not force irrelevant retrieved context into the answer. "
    "If the available evidence is insufficient, clearly state the limitation "
    "instead of presenting an unsupported claim as fact."
)
        user_prompt = f"""
RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer helpfully. Use the documents where relevant and general
knowledge where the documents are insufficient. Cite document-supported
claims with [filename.pdf, Page X]. If you go beyond the documents,
say that the additional information is from general knowledge.
"""
    else:
        system_prompt = """
You are a helpful Hybrid RAG AI Assistant.

No sufficiently relevant uploaded-document context was found for this
question. Answer using your general knowledge.

Do not pretend the answer came from the uploaded documents.
If the user asks something requiring current verification that you
cannot perform, clearly state the limitation.
"""
        user_prompt = question

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content or "",
        "sources": results,
        "mode": "hybrid",
    }


if __name__ == "__main__":
    stats = build_vector_store()
    print(stats)

    while True:
        question = input("\nAsk a question (or type exit): ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break

        result = generate_answer(question, "Hybrid")
        print("\nANSWER:\n")
        print(result["answer"])

        if result["sources"]:
            print("\nRETRIEVED SOURCES:")
            for source in result["sources"]:
                print(
                    f"- {source['source']} | Page {source['page']} | "
                    f"Similarity {source['score']:.3f}"
                )
