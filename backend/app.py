import os
import io
import json
from typing import List

import numpy as np
import httpx
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

import auth_utils
from groq import Groq as GroqClient
from openai import OpenAI as OpenAIClient

# ----------------------
# INIT
# ----------------------
load_dotenv()
auth_utils.init_db()

# ----------------------
# STREAMLIT CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ----------------------
st.set_page_config(
    page_title="RecallX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# SESSION STATE DEFAULTS
# ----------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to RecallX — drop a file and ask anything about it."}
    ]
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embs" not in st.session_state:
    st.session_state.embs = np.empty((0,384), dtype=np.float32)
if "sources" not in st.session_state:
    st.session_state.sources = []
if "show_settings" not in st.session_state:
    st.session_state.show_settings = os.getenv("SHOW_SETTINGS", "0") == "1"
if "mode" not in st.session_state:
    st.session_state.mode = "default"
if "personal_responses" not in st.session_state:
    st.session_state.personal_responses = {}

# ----------------------
# SIDEBAR LOGIN / SIGNUP
# ----------------------
st.sidebar.markdown("## 🔐 User Login / Signup")
auth_mode = st.sidebar.radio("Mode", ["Login", "Signup"])
username = st.sidebar.text_input("Username", key="auth_username")
password = st.sidebar.text_input("Password", type="password", key="auth_password")

if auth_mode == "Signup" and st.sidebar.button("Create Account"):
    if username and password:
        success = auth_utils.create_user(username, password)
        if success:
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

# Only allow access if logged in
if not st.session_state.user:
    st.warning("Please login to access RecallX features.")
    st.stop()

# ----------------------
# CSS INJECTION
# ----------------------
st.markdown(
    """
    <style>
    /* Minimalist theme */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background: #0b1220; color: #e6e6f0; }
    .block-container { max-width: 900px; margin: auto; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# UTILITY FUNCTIONS
# ----------------------
@st.cache_resource(show_spinner=False)
def get_model(name: str):
    return SentenceTransformer(name, trust_remote_code=True)

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embs.astype(np.float32)

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
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
    return chunks

def top_k_cosine(query_emb: np.ndarray, doc_embs: np.ndarray, k: int = 5) -> List[int]:
    if doc_embs.shape[0] == 0: return []
    sims = (doc_embs @ query_emb.reshape(-1,1)).ravel()
    return np.argsort(-sims)[:k].tolist()

# ----------------------
# FILE UPLOADER
# ----------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF or TXT)", type=["pdf","txt"], accept_multiple_files=True
)

if uploaded_files:
    st.session_state.chunks = []
    st.session_state.embs = np.empty((0,384), dtype=np.float32)
    st.session_state.sources = []
    for f in uploaded_files:
        data = f.read()
        ext = os.path.splitext(f.name)[1].lower()
        text = read_pdf(data) if ext == ".pdf" else read_txt(data)
        parts = chunk_text(text)
        new_chunks = [{"id": f"{f.name}:{i}", "text": p, "source": f.name} for i,p in enumerate(parts)]
        new_embs = embed_texts([c["text"] for c in new_chunks])
        if st.session_state.embs.size == 0:
            st.session_state.embs = new_embs
        else:
            st.session_state.embs = np.vstack([st.session_state.embs, new_embs])
        st.session_state.chunks.extend(new_chunks)
        st.session_state.sources.append(f.name)
    st.success(f"Loaded {len(st.session_state.chunks)} document chunks!")

# ----------------------
# CHAT DISPLAY
# ----------------------
for msg in st.session_state.messages:
    side = "right" if msg["role"]=="user" else "left"
    st.markdown(f"<div style='text-align:{side};'>{msg['content']}</div>", unsafe_allow_html=True)

# ----------------------
# USER INPUT FORM
# ----------------------
with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input("Ask anything...")
    submitted = st.form_submit_button("Send")

if submitted and user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    # --- Simple echo for now; can integrate LLM here ---
    st.session_state.messages.append({"role":"assistant","content":f"You asked: {user_q}"})
   st.rerun()
