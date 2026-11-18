import streamlit as st
from src.document_processor import process_uploaded_file
from src.chunking import create_chunks
from src.embeddings import generate_embeddings
from src.vector_store import store_chunks
from src.retrieval import retrieve_relevant_chunks
from src.chatbot import generate_answer

st.set_page_config(
    page_title="Medical RAG Experiment",
    layout="wide"
)

st.title("Medical RAG - Response Generation Experiment")
st.write(
    "Upload a clinical document to extract, chunk, "
    "embed and store its content."
)

st.sidebar.header("Processing Settings")

chunk_size = st.sidebar.slider(
    "Chunk size (characters)",
    min_value=200,
    max_value=1500,
    value=500,
    step=100
)

overlap = st.sidebar.slider(
    "Chunk overlap (characters)",
    min_value=0,
    max_value=400,
    value=100,
    step=50
)

uploaded_file = st.file_uploader(
    "Upload a clinical document",
    type=["pdf", "docx", "txt", "html", "htm"]
)

if uploaded_file:

    st.subheader("Document Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Filename", uploaded_file.name)
    col2.metric("File size", f"{uploaded_file.size / 1024:.2f} KB")
    col3.metric("Format", uploaded_file.name.split(".")[-1].upper())

    if st.button("Process Document", type="primary"):

        try:
            with st.spinner("Extracting document text..."):

                documents = process_uploaded_file(uploaded_file)

            st.session_state["documents"] = documents

            with st.spinner("Creating chunks..."):

                chunks = create_chunks(
                    documents,
                    chunk_size=chunk_size,
                    overlap=overlap
                )

            st.session_state["chunks"] = chunks

            with st.spinner("Generating embeddings..."):

                embeddings = generate_embeddings(chunks)

            st.session_state["embeddings"] = embeddings

            with st.spinner("Storing data in ChromaDB..."):

                collection = store_chunks(chunks, embeddings)
                st.session_state["collection"] = collection

            st.session_state["collection_count"] = collection.count()

            st.success("Document processed successfully!")

        except Exception as error:
            st.error(f"Processing failed: {error}")


if "documents" in st.session_state:

    documents = st.session_state["documents"]
    chunks = st.session_state["chunks"]
    embeddings = st.session_state["embeddings"]

    st.divider()
    st.header("1. Extraction Results")

    st.metric("Extracted documents", len(documents))

    for index, document in enumerate(documents):

        with st.expander(f"Extracted content {index + 1}"):

            st.write("Metadata")
            st.json(document["metadata"])

            st.write("Extracted text")
            st.text_area(
                "Content",
                document["text"],
                height=250,
                key=f"document_{index}"
            )

    st.divider()
    st.header("2. Chunking Results")

    col1, col2 = st.columns(2)

    col1.metric("Total chunks", len(chunks))
    col2.metric("Chunk size", chunk_size)

    for index, chunk in enumerate(chunks):

        with st.expander(f"Chunk {index + 1}"):

            st.write("Metadata")
            st.json(chunk["metadata"])

            st.write("Characters:", len(chunk["text"]))
            st.text(chunk["text"])

    st.divider()
    st.header("3. Embedding Results")

    st.metric("Embedding dimensions", embeddings.shape[1])
    st.metric("Total embeddings", embeddings.shape[0])

    st.write("First embedding preview:")
    st.write(embeddings[0][:10])

    st.divider()
    st.header("4. ChromaDB Storage")

    st.success("Data stored in the experiment database.")

    st.metric(
        "Total records in collection",
        st.session_state["collection_count"]
    )


# ==========================================
# 5. QUESTION ANSWERING
# ==========================================

st.divider()
st.header("5. Ask Questions About Your Document")

if "collection" in st.session_state:

    question = st.text_input(
        "Enter your medical question",
        placeholder="e.g. What are the symptoms of diabetes?"
    )

    if st.button("Generate Answer"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:

            with st.spinner("Retrieving relevant passages..."):

                retrieved_chunks = retrieve_relevant_chunks(
                    question,
                    st.session_state["collection"],
                    top_k=10,
                    threshold=0.0
                )

            st.subheader("Retrieved Passages")

            if retrieved_chunks:

                for index, chunk in enumerate(retrieved_chunks, 1):

                    with st.expander(f"Passage {index}"):

                        st.write(
                            "Similarity:",
                            round(chunk["similarity"], 4)
                        )

                        st.write("Metadata:")
                        st.json(chunk["metadata"])

                        st.write(chunk["text"])

            else:
                st.warning("No relevant passages found.")

            with st.spinner("Generating response..."):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            st.subheader("Generated Answer")
            st.write(answer)

else:
    st.info("Upload and process a document before asking questions.")
