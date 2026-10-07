import streamlit as st
from pathlib import Path

from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 RAG Document Assistant")
st.write("Upload a document and ask questions using AI.")


@st.cache_resource
def load_rag():

    return RAGPipeline(
        model_name="google/flan-t5-base",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3
    )


rag = load_rag()


uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"]
)


if uploaded_file:

    file_path = Path(uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            rag.ingest_file(str(file_path))

        st.session_state["document_ready"] = True

        st.success("Document processed successfully!")


if st.session_state.get("document_ready"):

    question = st.text_input(
        "Ask a question about the document"
    )

    if st.button("Ask AI") and question:

        with st.spinner("Generating answer..."):

            result = rag.ask(question)

        st.subheader("Answer")

        st.write(result["answer"])

        if result["sources"]:

            st.subheader("Sources")

            sources = set()

            for document in result["sources"]:

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                sources.add(source)

            for source in sources:
                st.write(f"📄 {source}")