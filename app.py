import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import ollama
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ---------------- PAGE TITLE ----------------

st.title("AI PDF Chatbot")

# ---------------- FILE UPLOAD ----------------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("Settings")

    st.write("Current Model:")
    st.write("phi3:mini")

    st.write("Embedding Model:")
    st.write("all-MiniLM-L6-v2")

    st.write("Total Uploaded PDFs:")
    st.write(
        len(uploaded_files)
        if uploaded_files
        else 0
    )

    st.write("Total Chunks:")

    st.write(
        len(st.session_state.chunks)
        if "chunks" in st.session_state
        else 0
    )

    st.divider()

    st.subheader("Retrieved Sources")

    if "latest_sources" in st.session_state:

        for source, chunk in zip(
            st.session_state.latest_sources,
            st.session_state.latest_chunks
        ):

            st.markdown(f"### {source}")

            with st.expander("View Retrieved Chunk"):

                st.write(chunk)

# ---------------- CLEAR CHAT ----------------

if st.button("Clear Chat"):

    st.session_state.messages = []

    st.session_state.latest_chunks = []

    st.session_state.latest_sources = []

# ---------------- PROCESS PDFs ----------------

if uploaded_files:

    # Build embeddings only once
    if "faiss_index" not in st.session_state:

        all_chunks = []

        chunk_sources = []

        # Read all PDFs
        for uploaded_file in uploaded_files:

            pdf_reader = PdfReader(uploaded_file)

            text = ""

            # Extract text
            for page in pdf_reader.pages:

                extracted_text = page.extract_text()

                if extracted_text:

                    text += extracted_text

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_text(text)

            # Save chunks + source names
            for chunk in chunks:

                all_chunks.append(chunk)

                chunk_sources.append(
                    uploaded_file.name
                )

        # Save chunks
        st.session_state.chunks = all_chunks

        st.session_state.chunk_sources = chunk_sources

        # Load embedding model
        embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

        st.session_state.embedding_model = embedding_model

        # Create embeddings
        embeddings = embedding_model.encode(
            all_chunks
        )

        embeddings = np.array(
            embeddings
        ).astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        # Save index
        st.session_state.faiss_index = index

        st.success(
            "Embeddings created successfully!"
        )

    # ---------------- REUSE OBJECTS ----------------

    chunks = st.session_state.chunks

    chunk_sources = st.session_state.chunk_sources

    embedding_model = st.session_state.embedding_model

    index = st.session_state.faiss_index

    st.write("Total Chunks:", len(chunks))

    # ---------------- CHAT HISTORY ----------------

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

    # ---------------- USER INPUT ----------------

    user_question = st.chat_input(
        "Ask a question about the PDFs"
    )

    if user_question:

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # Show user message instantly
        with st.chat_message("user"):

            st.write(user_question)

        # ---------------- RETRIEVAL QUERY ----------------

        retrieval_query = user_question

        previous_user_messages = [
            msg["content"]
            for msg in st.session_state.messages
            if msg["role"] == "user"
        ]

        if len(previous_user_messages) >= 2:

            retrieval_query = (
                previous_user_messages[-2]
                + " "
                + user_question
            )

        # ---------------- QUESTION EMBEDDING ----------------

        question_embedding = embedding_model.encode(
            [retrieval_query]
        )

        question_embedding = np.array(
            question_embedding
        ).astype('float32')

        # ---------------- VECTOR SEARCH ----------------

        k = 8

        distances, indices = index.search(
            question_embedding,
            k
        )

        # ---------------- RETRIEVE CHUNKS ----------------

        relevant_chunks = []

        relevant_sources = []

        for i in indices[0]:

            relevant_chunks.append(
                chunks[i]
            )

            relevant_sources.append(
                chunk_sources[i]
            )

        # Save retrieved chunks for sidebar
        st.session_state.latest_chunks = relevant_chunks

        st.session_state.latest_sources = relevant_sources

        # ---------------- BUILD CONTEXT ----------------

        context = ""

        for source, chunk in zip(
            relevant_sources,
            relevant_chunks
        ):

            context += f"""
            Source: {source}

            Content:
            {chunk}

            """

        # ---------------- CONVERSATION HISTORY ----------------

        conversation_history = ""

        for message in st.session_state.messages[-4:]:

            conversation_history += f"""
            {message['role']}:
            {message['content']}
            """

        # ---------------- PROMPT ----------------

        prompt = f"""
        You are an AI assistant for answering questions about uploaded PDFs.

        Instructions:
        - Use ONLY the provided context.
        - Mention source document names in your answer whenever relevant.
        - If information comes from multiple documents, clearly mention each document.
        - If the answer is not found clearly, say:
          "I could not find a clear answer in the uploaded documents."
        - Be concise but informative.
        - Do NOT make up information.

        Conversation History:
        {conversation_history}

        Retrieved Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        # ---------------- GENERATE RESPONSE ----------------

        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                response = ollama.chat(
                    model='phi3:mini',
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ],
                    stream=True
                )

                answer = ""

                response_placeholder = st.empty()

                for chunk in response:

                    content = chunk['message']['content']

                    answer += content

                    response_placeholder.markdown(
                        answer + "▌"
                    )

                # Final clean response
                response_placeholder.markdown(answer)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        }) 