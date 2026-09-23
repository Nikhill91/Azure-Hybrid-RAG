import json
import re
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from rag import (
    DOCUMENTS_FOLDER,
    MANIFEST_PATH,
    VECTOR_STORE_FOLDER,
    build_vector_store,
    generate_answer,
    get_configuration_status,
    knowledge_base_stats,
    vector_store_exists,
)


APP_VERSION = "1.0.0"
ANSWER_MODES = ["Hybrid", "Documents Only", "General Knowledge"]

MAX_UPLOAD_SIZE_MB = 25
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

MODE_DESCRIPTIONS = {
    "Hybrid": "Ground answers in your documents first, then add general context when useful.",
    "Documents Only": "Use only retrieved document context. Best for audit-friendly answers.",
    "General Knowledge": "Skip retrieval and use the configured model as a general assistant.",
}

SUGGESTIONS = [
    "Summarize the key points in the knowledge base",
    "What risks or open questions are mentioned?",
    "Create an executive brief with citations",
]


def safe_filename(filename: str) -> str:
    """Keep uploads inside the knowledge-base directory and avoid collisions."""
    cleaned = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", cleaned).strip(" .")

    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"

    return cleaned or "uploaded-document.pdf"


def clear_knowledge_base() -> None:
    DOCUMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_FOLDER.mkdir(parents=True, exist_ok=True)

    for path in DOCUMENTS_FOLDER.iterdir():
        if path.is_file():
            path.unlink()

    for path in (
        VECTOR_STORE_FOLDER / "index.faiss",
        VECTOR_STORE_FOLDER / "chunks.pkl",
        MANIFEST_PATH,
    ):
        if path.exists():
            path.unlink()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(
        f"{len(sources)} retrieved source(s)",
        icon=":material/menu_book:",
    ):
        for source in sources:
            with st.container(border=True):
                source_col, score_col = st.columns(
                    [4, 1],
                    vertical_alignment="center",
                )

                with source_col:
                    st.markdown(
                        f":material/description: **{source['source']}**"
                    )
                    st.caption(f"Page {source['page']}")

                with score_col:
                    st.metric("Match", f"{source['score']:.0%}")

                st.caption(
                    source["text"][:280]
                    + ("..." if len(source["text"]) > 280 else "")
                )


def render_answer_metadata(
    mode: str,
    has_sources: bool,
    latency_ms: int | None = None,
) -> None:
    descriptions = {
        "general": ":material/public: General knowledge response.",
        "hybrid": (
            ":material/merge_type: Grounded with document context and broader knowledge."
            if has_sources
            else ":material/public: No relevant document context found; broader knowledge used."
        ),
        "documents": ":material/article: Strictly grounded in uploaded documents.",
    }

    metadata = descriptions.get(mode, "")

    if latency_ms is not None:
        metadata += f"  ·  {latency_ms / 1000:.1f}s"

    st.caption(metadata)


def set_pending_question(question: str) -> None:
    st.session_state.pending_question = question


def export_conversation() -> str:
    return json.dumps(
        {
            "product": "Azure Hybrid RAG",
            "exported_at": st.session_state.get("exported_at", "session"),
            "answer_mode": st.session_state.get("answer_mode", "Hybrid"),
            "messages": st.session_state.messages,
        },
        indent=2,
    )


st.set_page_config(
    page_title="Azure Hybrid RAG",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DOCUMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_FOLDER.mkdir(parents=True, exist_ok=True)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("kb_ready", vector_store_exists())
st.session_state.setdefault("pending_question", None)

config = get_configuration_status()
stats = knowledge_base_stats()

with st.sidebar:
    st.markdown("## :material/auto_awesome: Azure Hybrid RAG")
    st.caption(
        "A dependable workspace for grounded research across your PDF knowledge base."
    )

    if config["ready"]:
        st.badge(
            "Azure connected",
            icon=":material/cloud_done:",
            color="green",
        )
    else:
        st.badge(
            "Configuration needed",
            icon=":material/cloud_off:",
            color="orange",
        )
        st.caption(f"Missing: {config['missing']}")

    st.space("small")

    st.subheader("Knowledge base", divider="gray")

    uploaded_files = st.file_uploader(
        "Add source documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs with selectable text are fastest. Scanned PDFs use local Tesseract OCR.",
    )

    # Person 4 contribution:
    # Validate uploaded PDF sizes before allowing processing.
    if uploaded_files:
        oversized_files = [
            file.name
            for file in uploaded_files
            if file.size > MAX_UPLOAD_SIZE_BYTES
        ]

        with st.container(border=True):
            total_mb = sum(
                file.size for file in uploaded_files
            ) / (1024 * 1024)

            if oversized_files:
                st.error(
                    f"{len(oversized_files)} file(s) exceed the "
                    f"{MAX_UPLOAD_SIZE_MB} MB per-file limit.",
                    icon=":material/error:",
                )

                for filename in oversized_files:
                    st.caption(
                        f":material/block: {filename}"
                    )
            else:
                st.markdown(
                    f"**{len(uploaded_files)} PDF(s) ready**"
                )

                st.caption(
                    f"{total_mb:.1f} MB · processing replaces the current index"
                )

            with st.expander(
                "Review files",
                icon=":material/attach_file:",
            ):
                for file in uploaded_files:
                    size_mb = file.size / (1024 * 1024)

                    st.caption(
                        f":material/picture_as_pdf: "
                        f"{file.name} · {size_mb:.1f} MB"
                    )

    process = st.button(
        "Process documents",
        type="primary",
        icon=":material/rocket_launch:",
        width="stretch",
        disabled=(
            not uploaded_files
            or not config["ready"]
            or any(
                file.size > MAX_UPLOAD_SIZE_BYTES
                for file in uploaded_files
            )
        ),
    )

    clear_kb = st.button(
        "Clear knowledge base",
        icon=":material/delete_sweep:",
        width="stretch",
        disabled=not st.session_state.kb_ready,
    )

    st.space("small")

    st.subheader("Answer mode", divider="gray")

    answer_mode = st.segmented_control(
        "Choose how answers should be generated",
        ANSWER_MODES,
        default="Hybrid",
        key="answer_mode",
        width="stretch",
    ) or "Hybrid"

    st.caption(MODE_DESCRIPTIONS[answer_mode])

    st.space("small")

    st.subheader("Session", divider="gray")

    clear_chat = st.button(
        "Clear conversation",
        icon=":material/delete_outline:",
        width="stretch",
        disabled=not st.session_state.messages,
    )

    st.download_button(
        "Export conversation",
        data=export_conversation(),
        file_name="azure-hybrid-rag-session.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
        disabled=not st.session_state.messages,
    )

    with st.expander(
        "Runtime details",
        icon=":material/settings:",
    ):
        st.caption(f"App version {APP_VERSION}")
        st.caption(f"LLM: `{config['llm_model']}`")
        st.caption(
            f"Embeddings: `{config['embedding_model']}`"
        )
        st.caption(
            f"Endpoint: "
            f"`{str(config['endpoint']).replace('https://', '').split('/')[0]}`"
        )


if clear_chat:
    st.session_state.messages = []
    st.rerun()


if clear_kb:
    clear_knowledge_base()
    st.session_state.kb_ready = False
    st.session_state.messages = []
    st.rerun()


if process:
    try:
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent,
            prefix=".ingest-",
        ) as staging_dir:

            staging_path = Path(staging_dir)

            with st.status(
                "Building your knowledge base...",
                expanded=True,
            ) as status:

                used_names = set()

                for uploaded_file in uploaded_files:
                    name = safe_filename(uploaded_file.name)

                    if name in used_names:
                        stem = Path(name).stem
                        name = (
                            f"{stem}-{len(used_names) + 1}.pdf"
                        )

                    used_names.add(name)

                    (
                        staging_path / name
                    ).write_bytes(
                        uploaded_file.getbuffer()
                    )

                    st.write(f"Staged `{name}`")

                st.write(
                    "Extracting text, applying OCR where needed, "
                    "and creating embeddings..."
                )

                new_stats = build_vector_store(
                    staging_path
                )

                for old_file in DOCUMENTS_FOLDER.iterdir():
                    if old_file.is_file():
                        old_file.unlink()

                for staged_file in staging_path.iterdir():
                    shutil.copy2(
                        staged_file,
                        DOCUMENTS_FOLDER / staged_file.name,
                    )

                status.update(
                    label="Knowledge base published",
                    state="complete",
                    expanded=False,
                )

        st.session_state.kb_ready = True
        st.session_state.messages = []

        st.success(
            f"Indexed {new_stats['files']} file(s), "
            f"{new_stats['pages']} page(s), and "
            f"{new_stats['chunks']} searchable chunks.",
            icon=":material/check_circle:",
        )

        stats = new_stats

    except Exception as exc:
        st.error(
            f"Ingestion failed: {exc}",
            icon=":material/error:",
        )


with st.container(width=1100):

    top_col, status_col = st.columns(
        [5, 1],
        vertical_alignment="bottom",
    )

    with top_col:
        st.title(
            "Research workspace",
            anchor=False,
        )

        st.caption(
            "Ask precise questions, keep answers grounded, "
            "and trace every document-backed claim."
        )

    with status_col:
        if st.session_state.kb_ready:
            st.badge(
                "Knowledge base ready",
                icon=":material/check_circle:",
                color="green",
            )
        else:
            st.badge(
                "No index yet",
                icon=":material/pending:",
                color="orange",
            )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Documents",
        stats.get("files", 0),
        border=True,
    )

    metric_cols[1].metric(
        "Pages",
        stats.get("pages", 0),
        border=True,
    )

    metric_cols[2].metric(
        "Searchable chunks",
        stats.get("chunks", 0),
        border=True,
    )

    metric_cols[3].metric(
        "Mode",
        answer_mode,
        border=True,
    )

    if not config["ready"]:
        st.warning(
            "Connect Azure OpenAI to enable answers. "
            "Add `AZURE_API_KEY` and `openai_endpoint` "
            "to your local environment.",
            icon=":material/key:",
        )

    elif (
        not st.session_state.kb_ready
        and answer_mode != "General Knowledge"
    ):
        st.info(
            "Upload and process at least one PDF to enable "
            "grounded modes. General Knowledge remains available "
            "without an index.",
            icon=":material/upload_file:",
        )

    elif st.session_state.kb_ready:
        documents = stats.get("documents", [])

        with st.container(border=True):
            source_col, update_col = st.columns(
                [4, 1],
                vertical_alignment="center",
            )

            with source_col:
                st.markdown("**Active source set**")

                st.caption(
                    ", ".join(documents[:4])
                    + (
                        f" and {len(documents) - 4} more"
                        if len(documents) > 4
                        else ""
                    )
                )

            with update_col:
                st.caption(
                    f"Updated {stats.get('created_at', 'recently')[:10]}"
                )

    st.space("small")

    if not st.session_state.messages:
        with st.container(border=True):
            st.subheader(
                "Start with a focused question",
                icon=":material/search:",
            )

            st.write(
                "Use the prompts below or ask in your own words. "
                "Answers show their retrieval mode, latency, "
                "and supporting passages."
            )

            with st.container(horizontal=True):
                for index, suggestion in enumerate(SUGGESTIONS):
                    st.button(
                        suggestion,
                        key=f"suggestion_{index}",
                        on_click=set_pending_question,
                        args=(suggestion,),
                    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_answer_metadata(
                    message.get("mode", ""),
                    bool(message.get("sources")),
                    message.get("latency_ms"),
                )

                render_sources(
                    message.get("sources", [])
                )

    question = st.chat_input(
        "Ask about your sources..."
        if answer_mode != "General Knowledge"
        else "Ask a general question...",
        disabled=(
            not config["ready"]
            or (
                not st.session_state.kb_ready
                and answer_mode != "General Knowledge"
            )
        ),
    )

    question = (
        question
        or st.session_state.pop(
            "pending_question",
            None,
        )
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching your sources and composing an answer..."
                ):
                    result = generate_answer(
                        question,
                        mode=answer_mode,
                    )

                st.markdown(
                    result["answer"]
                )

                render_answer_metadata(
                    result["mode"],
                    bool(result["sources"]),
                    result.get("latency_ms"),
                )

                render_sources(
                    result["sources"]
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "mode": result["mode"],
                        "latency_ms": result.get("latency_ms"),
                    }
                )

            except Exception as exc:
                st.error(
                    f"Unable to answer this question: {exc}",
                    icon=":material/error:",
                )

    with st.expander(
        "Knowledge base details",
        icon=":material/insights:",
    ):
        if stats.get("documents"):
            st.dataframe(
                [
                    {
                        "Document": document,
                        "Status": "Indexed",
                    }
                    for document in stats["documents"]
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption(
                "No documents are indexed yet."
            )