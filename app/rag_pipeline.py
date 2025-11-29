from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import faiss
from pypdf import PdfReader
import tiktoken
import streamlit as st
import google.generativeai as genai


# ---------------- CONFIG ----------------

@dataclass
class RAGConfig:
    # Gemini embedding model
    embedding_model: str = "models/embedding-001" 
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50


# ---------------- CLIENT HELPER ----------------

def _configure_gemini():
    """Configures the Gemini client using available secrets."""
    try:
        # Check if [google] section exists, otherwise try root level or [gemini]
        if "google" in st.secrets:
            api_key = st.secrets["google"]["api_key"]
        elif "gemini" in st.secrets:
            api_key = st.secrets["gemini"]["api_key"]
        else:
            api_key = st.secrets.get("google_api_key", "")
            
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")


# ---------------- VECTOR STORE ----------------

class RAGStore:
    """FAISS vector DB with embeddings & metadata."""

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.docs: List[Dict[str, Any]] = []
        self.dim = dim

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        if embeddings.size == 0:
            return
        self.index.add(embeddings.astype("float32"))
        self.docs.extend(metadatas)

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4):
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.docs):
                results.append(self.docs[idx])
        return results


# ---------------- PDF PARSING ----------------

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


# ---------------- CHUNKING ----------------

def _chunk_text(text: str, max_tokens=500, overlap_tokens=50, encoding_name="cl100k_base"):
    # We use tiktoken for rough token counting even for Gemini, as it's a good approximation
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        chunks.append(chunk_text)
        start += max_tokens - overlap_tokens

    return chunks


# ---------------- BUILD RAG STORE ----------------

def build_rag_store_from_uploads(uploaded_files, cfg: RAGConfig | None = None):
    if cfg is None:
        cfg = RAGConfig()

    _configure_gemini()
    all_chunks = []
    texts = []

    # Extract and chunk PDFs
    for file in uploaded_files:
        try:
            raw = file.read()
            text = _extract_text_from_pdf(raw)
        except Exception as e:
            st.warning(f"Error reading {file.name}: {e}")
            continue

        if not text.strip():
            continue

        chunks = _chunk_text(
            text,
            max_tokens=cfg.chunk_size_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
        )

        for i, ch in enumerate(chunks):
            all_chunks.append({
                "content": ch,
                "source": file.name,
                "chunk_id": i,
            })
            texts.append(ch)

    if len(texts) == 0:
        # 768 is the dimension for Gemini embedding-001
        return RAGStore(dim=768), []

    # Embed all chunks
    embeddings = []
    batch_size = 20  # Gemini batch size limit is smaller than OpenAI's sometimes

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Gemini Embedding Call
            result = genai.embed_content(
                model=cfg.embedding_model,
                content=batch,
                task_type="retrieval_document"
            )
            # result['embedding'] is a list of embeddings
            emb = result['embedding']
            embeddings.extend(emb)
        except Exception as e:
            st.error(f"Error embedding batch: {e}")
            continue

    if not embeddings:
        return RAGStore(dim=768), []

    emb_array = np.array(embeddings, dtype="float32")
    dim = emb_array.shape[1]

    store = RAGStore(dim=dim)
    store.add_embeddings(emb_array, all_chunks)

    return store, all_chunks


# ---------------- QUERY EMBEDDING ----------------

def embed_query(text: str, cfg: RAGConfig | None = None) -> np.ndarray:
    if cfg is None:
        cfg = RAGConfig()

    _configure_gemini()
    
    result = genai.embed_content(
        model=cfg.embedding_model,
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result['embedding'], dtype="float32")


# ---------------- RAG TOOL (GENERATION) ----------------

def rag_tool(store: RAGStore, question: str) -> str:
    """
    Retrieves context from the store and generates an answer using Gemini.
    """
    _configure_gemini()

    # 1. Embed the user's question
    query_emb = embed_query(question)
    
    # 2. Search for relevant chunks (Top 4)
    results = store.similarity_search(query_emb, k=4)
    
    if not results:
        return "I couldn't find any information about that in the uploaded documents."
    
    # 3. Construct context from results
    context_text = "\n\n---\n\n".join([doc["content"] for doc in results])
    
    # 4. Generate answer using Gemini
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    system_prompt = (
        "You are a helpful hotel booking assistant. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer is not in the context, say you don't know."
    )
    
    user_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {question}"
    
    try:
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"