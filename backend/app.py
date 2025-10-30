import os, io, json
from typing import List
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import auth_utils
from dotenv import load_dotenv
from groq import Groq as GroqClient
from openai import OpenAI as OpenAIClient
import httpx

# --- AUTH SETUP ---
auth_utils.init_db()
if "user" not in st.session_state:
    st.session_state.user = None

st.sidebar.markdown("## 🔐 User Login / Signup")
auth_mode = st.sidebar.radio("Mode", ["Login", "Signup"])
username = st.sidebar.text_input("Username", key="auth_username")
password = st.sidebar.text_input("Password", type="password", key="auth_password")

if auth_mode == "Signup" and st.sidebar.button("Create Account"):
    if username and password:
        if auth_utils.create_user(username, password):
            st.success("Account created! Please log in.")
        else:
            st.error("Username already exists.")
    else:
        st.warning("Enter username and password.")

elif auth_mode == "Login" and st.sidebar.button("Login"):
    if auth_utils.authenticate(username, password):
        st.session_state.user = username
        st.success(f"Logged in as {username}")
    else:
        st.error("Invalid credentials")

if not st.session_state.user:
    st.warning("Please login to access RecallX features.")
    st.stop()

# --- ENV VARS & SETTINGS ---
load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

st.set_page_config(
    page_title="RecallX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- UTILITY FUNCTIONS ---
@st.cache_resource
def get_model(name: str):
    return SentenceTransformer(name, trust_remote_code=True)

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model(EMBED_MODEL)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")

def chunk_text(text: str, max_chars=1000, overlap=150) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf: chunks.append(buf)
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    chunks.append(p[start:end])
                    start = end - overlap if end < len(p) else end
            else:
                buf = p
    if buf: chunks.append(buf)
    final = []
    for i, c in enumerate(chunks):
        if i == 0: final.append(c)
        else:
            tail = final[-1][-overlap:] if overlap > 0 else ""
            final.append((tail + " " + c).strip())
    return final

def top_k_cosine(query_emb: np.ndarray, doc_embs: np.ndarray, k: int = 5) -> List[int]:
    if doc_embs.shape[0] == 0: return []
    sims = (doc_embs @ query_emb.reshape(-1,1)).ravel()
    return np.argsort(-sims)[:k].tolist()

def llm_answer(provider, model, api_key, question, context) -> str:
    system_msg = (
        "You are a precise assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say: 'I couldn’t find this in the document.'"
    )
    user_msg = f"Question: {question}\n\nContext:\n{context}\n"
    if provider == "groq":
        client = GroqClient(api_key=api_key, http_client=httpx.Client(timeout=60, trust_env=False))
        resp = client.chat.completions.create(model=model, messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}], temperature=0.2, max_tokens=350)
        return resp.choices[0].message.content.strip()
    elif provider == "openai":
        client = OpenAIClient(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}], temperature=0.2, max_tokens=350)
        return resp.choices[0].message.content.strip()
    else:
        return "Unknown provider"

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Welcome to RecallX — drop a file and ask anything about it."}]
if "chunks" not in st.session_state: st.session_state.chunks = []
if "embs" not in st.session_state: st.session_state.embs = np.empty((0,384), dtype=np.float32)
if "sources" not in st.session_state: st.session_state.sources = []

# --- FILE UPLOADER ---
uploaded_files = st.file_uploader("Upload document (PDF or TXT)", type=["pdf","txt"], accept_multiple_files=True)
if uploaded_files:
    st.session_state.chunks = []
    st.session_state.embs = np.empty((0,384), dtype=np.float32)
    st.session_state.sources = []
    for f in uploaded_files:
        data = f.read()
        text = read_pdf(data) if f.name.lower().endswith(".pdf") else read_txt(data)
        parts = chunk_text(text)
        new_chunks = [{"id": f"{f.name}:{i}", "text": p, "source": f.name} for i,p in enumerate(parts)]
        embs = embed_texts([c["text"] for c in new_chunks])
        if st.session_state.embs.size == 0:
            st.session_state.embs = embs
        else:
            st.session_state.embs = np.vstack([st.session_state.embs, embs])
        st.session_state.chunks.extend(new_chunks)
        st.session_state.sources.append(f.name)
    st.success(f"Loaded {len(st.session_state.chunks)} document chunks!")

# --- CHAT AREA ---
for m in st.session_state.messages:
    side = "rx-right" if m["role"]=="user" else "rx-left"
    st.markdown(f'<div class="rx-row {side}"><div class="rx-bubble">{m["content"]}</div></div>', unsafe_allow_html=True)

# --- INPUT FORM ---
with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input("Ask anything...", "", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")
if submitted and user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    # RAG QUERY
    if len(st.session_state.chunks) == 0:
        answer = "No documents loaded yet."
    else:
        q_emb = embed_texts([user_q])[0]
        idxs = top_k_cosine(q_emb, st.session_state.embs, k=5)
        selected_chunks = [st.session_state.chunks[i] for i in idxs]
        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in selected_chunks)
        answer = llm_answer(DEFAULT_PROVIDER, DEFAULT_MODEL, os.getenv("GROQ_API_KEY",""), user_q, context)
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()
