import os
import shutil

import streamlit as st

from rag import build_vector_store, generate_answer, vector_store_exists


DOCUMENTS_FOLDER = "documents"
VECTOR_STORE_FOLDER = "vector_store"
ANSWER_MODES = ["Hybrid", "Documents Only", "General Knowledge"]


def render_sources(sources):
    """Render retrieved document context without overwhelming the answer."""
    if not sources:
        return

    with st.expander(
        f"{len(sources)} retrieved source(s)",
        icon=":material/menu_book:",
    ):
        for source in sources:
            with st.container(border=True):
                st.markdown(
                    f":material/description: **{source['source']}**  "
                    f"\nPage **{source['page']}** - "
                    f"Similarity **{source['score']:.3f}**"
                )


def render_answer_metadata(mode, has_sources):
    if mode == "general":
        st.caption(":material/public: Answered using general knowledge.")
    elif mode == "hybrid":
        if has_sources:
            st.caption(
                ":material/merge_type: Hybrid answer using document context "
                "and general knowledge as needed."
            )
        else:
            st.caption(
                ":material/public: No sufficiently relevant document context "
                "was found; answered using general knowledge."
            )
    elif mode == "documents":
        st.caption(":material/article: Answered from uploaded documents only.")


def clear_knowledge_base():
    for folder in [DOCUMENTS_FOLDER, VECTOR_STORE_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)


os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="Azure Hybrid RAG",
    page_icon=":material/auto_awesome:",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = vector_store_exists()

with st.sidebar:
    st.markdown("## :material/library_books: Knowledge base")
    st.caption("Upload PDFs to ground answers in your own content.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once. Scanned pages are OCR processed.",
    )

    if uploaded_files:
        with st.container(border=True):
            st.markdown(f"**{len(uploaded_files)} PDF(s) selected**")
            with st.expander("View selected files", icon=":material/attach_file:"):
                for file in uploaded_files:
                    st.caption(f":material/picture_as_pdf: {file.name}")

    process = st.button(
        "Process documents",
        type="primary",
        icon=":material/rocket_launch:",
        width="stretch",
    )
    clear = st.button(
        "Clear knowledge base",
        icon=":material/delete_sweep:",
        width="stretch",
    )

    st.space("small")
    st.subheader("Answer mode", divider="gray")
    answer_mode = st.segmented_control(
        "Choose how answers should be generated",
        ANSWER_MODES,
        default="Hybrid",
        key="answer_mode",
        width="stretch",
    )
    answer_mode = answer_mode or "Hybrid"

    mode_descriptions = {
        "Hybrid": "Uses relevant documents first, then broader model knowledge when useful.",
        "Documents Only": "Answers strictly from retrieved document context.",
        "General Knowledge": "Skips document retrieval and uses the general model.",
    }
    st.caption(mode_descriptions[answer_mode])

    with st.expander("Model configuration", icon=":material/settings:"):
        st.caption("Configured through your environment variables.")
        st.code(
            "LLM: gpt-5-mini\n"
            "Embedding: text-embedding-3-large",
            language="text",
        )

    st.caption("Azure Hybrid RAG")

if clear:
    clear_knowledge_base()
    st.session_state.kb_ready = False
    st.session_state.messages = []
    st.rerun()

if process:
    if not uploaded_files:
        st.warning(
            "Upload at least one PDF before processing.",
            icon=":material/upload_file:",
        )
    else:
        st.session_state.kb_ready = False
        try:
            for filename in os.listdir(DOCUMENTS_FOLDER):
                path = os.path.join(DOCUMENTS_FOLDER, filename)
                if os.path.isfile(path):
                    os.remove(path)

            # Remove the old index first so a failed rebuild cannot appear ready.
            for path in [
                os.path.join(VECTOR_STORE_FOLDER, "index.faiss"),
                os.path.join(VECTOR_STORE_FOLDER, "chunks.pkl"),
            ]:
                if os.path.exists(path):
                    os.remove(path)

            with st.status(
                "Processing documents...",
                expanded=True,
                state="running",
            ) as status:
                for file in uploaded_files:
                    path = os.path.join(DOCUMENTS_FOLDER, file.name)
                    with open(path, "wb") as output_file:
                        output_file.write(file.getbuffer())
                    st.write(f"Saved `{file.name}`")

                st.write("Extracting text and applying OCR where needed...")
                stats = build_vector_store()

                status.update(
                    label="Knowledge base ready",
                    state="complete",
                    expanded=False,
                )

            st.session_state.kb_ready = True
            st.session_state.messages = []
            st.success(
                f"Processed {stats['files']} file(s), {stats['pages']} page(s), "
                f"and {stats['chunks']} chunks.",
                icon=":material/check_circle:",
            )

        except Exception as exc:
            st.session_state.kb_ready = False
            st.error(f"Processing failed: {exc}", icon=":material/error:")

with st.container(width=960):
    st.title("Azure Hybrid RAG", icon=":material/auto_awesome:")
    st.caption(
        "A focused workspace for asking questions across your PDFs with Azure "
        "OpenAI, FAISS retrieval, and OCR support."
    )

    with st.container(border=True, gap="small"):
        header_col, status_col = st.columns([4, 1], vertical_alignment="center")
        with header_col:
            st.subheader("Research workspace")
            st.write(
                "Manage your sources from the sidebar, choose an answer mode, "
                "and keep the conversation here."
            )
        with status_col:
            if st.session_state.kb_ready:
                st.badge("Ready", icon=":material/check_circle:", color="green")
                st.caption("Index available")
            else:
                st.badge("Not indexed", icon=":material/pending:", color="orange")
                st.caption("General mode available")

    st.space("small")

    if not st.session_state.kb_ready and answer_mode != "General Knowledge":
        with st.container(border=True):
            st.subheader("Build your knowledge base", icon=":material/upload_file:")
            st.write(
                "Upload one or more PDFs and select **Process documents** to enable "
                "Hybrid and Documents Only answers."
            )
            st.caption(
                ":material/lightbulb: General Knowledge mode remains available "
                "without an uploaded document index."
            )
    elif st.session_state.kb_ready:
        st.caption(
            ":material/check_circle: Your uploaded documents are ready for "
            "grounded answers."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_answer_metadata(
                    message.get("mode", ""), bool(message.get("sources"))
                )
                render_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask about your PDFs or use general knowledge...",
        disabled=not st.session_state.kb_ready and answer_mode != "General Knowledge",
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and thinking..."):
                    result = generate_answer(question, mode=answer_mode)

                st.markdown(result["answer"])
                render_answer_metadata(result["mode"], bool(result["sources"]))
                render_sources(result["sources"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "mode": result["mode"],
                })

            except Exception as exc:
                st.error(f"RAG error: {exc}", icon=":material/error:")
