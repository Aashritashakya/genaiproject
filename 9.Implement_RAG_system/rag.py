import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# ----------------------
# Load models once
# ----------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return embedder, generator

embedder, generator = load_models()

# ----------------------
# Extract text from PDF
# ----------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ----------------------
# Split into chunks (basic)
# ----------------------
def split_into_chunks(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ----------------------
# Streamlit UI
# ----------------------
st.title("📄 PDF-based RAG App")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully.")
    raw_text = extract_text_from_pdf(uploaded_file)
    chunks = split_into_chunks(raw_text)

    st.markdown(f"**Document split into {len(chunks)} chunks.**")

    # Embed and build index
    doc_embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    # Query UI
    user_query = st.text_input("Ask a question based on your PDF:")

    if user_query:
        query_embedding = embedder.encode([user_query], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=3)
        retrieved_chunks = [chunks[i] for i in I[0]]

        st.subheader("📚 Retrieved Context")
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")

        context = " ".join(retrieved_chunks)
        prompt = f"Context: {context} Question: {user_query}"

        with st.spinner("Generating answer..."):
            result = generator(prompt, max_length=150, do_sample=False)
            st.subheader("🧠 Answer")
            st.write(result[0]['generated_text'])
