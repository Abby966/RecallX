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
from auth_utils import init_db  
init_db()
load_dotenv()
st.set_page_config(page_title="RecallX", page_icon="📦", layout="wide")

ping = st.experimental_get_query_params().get("ping")
if ping:
    st.write("pong")
    st.stop()  

# --- SESSION STATE INITIALIZATION (top of script) ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to RecallX — drop a file and ask anything about it."}]

if "sources" not in st.session_state:
    st.session_state["sources"] = []

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "embs" not in st.session_state:
    st.session_state["embs"] = np.empty((0,384), dtype=np.float32)

if "user" not in st.session_state:
    st.session_state["user"] = None

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# --- INJECT MODERN CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

:root {
    --rx-indigo: #4f46e5;
    --rx-indigo-600: #4338ca;
    --rx-sky: #0ea5e9;
    --rx-emerald: #10b981;
    --rx-white: #ffffff;
    --rx-text: #e6e6f0;
    --rx-text-dim: #b7bfd6;
    --rx-border: rgba(255,255,255,0.22);
    --rx-bg: radial-gradient(1200px 600px at 10% -10%, rgba(79,70,229,0.10) 0%, rgba(14,165,233,0.08) 40%, transparent 70%),
             radial-gradient(900px 500px at 110% 10%, rgba(16,185,129,0.12) 0%, transparent 60%),
             #0b1220;
}

.stApp {
    background: var(--rx-bg);
    background-attachment: fixed;
}

/* Center content container */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 4rem !important;
    max-width: 900px !important;
    margin-left: auto;
    margin-right: auto;
}

/* Header */
.rx-header-container {
    display:flex;
    justify-content: space-between;
    align-items:center;
    margin-bottom: 5px;
}
.rx-title { font-weight:700; font-size:1.6rem; color:var(--rx-white); }

/* Welcome Card */
.welcome-card {
    display:flex;
    justify-content:center;
    margin:15vh 0;
}
.welcome-card div {
    background: rgba(79,70,229,0.15);
    color:#fff;
    padding:25px 35px;
    border-radius:20px;
    box-shadow:0 6px 20px rgba(0,0,0,0.3);
    font-size:1.2rem;
    max-width:700px;
    text-align:center;
}

/* Document Chips */
.rx-chip {
    display:inline-block;
    background: rgba(79,70,229,0.15);
    color:#fff;
    padding:6px 12px;
    margin:4px;
    border-radius:999px;
    font-size:0.85rem;
}

/* Chat area */
.rx-chat { margin-top: 12px; }
.rx-row { margin: 12px 0; display:flex; align-items:flex-start; }

/* Bubbles */
.rx-bubble {
    display:inline-block;
    padding:14px 18px;
    max-width:80%;
    line-height:1.5;
    word-wrap:break-word;
    white-space:pre-wrap;
    font-size:1rem;
    border-radius:20px;
    opacity:0;
    animation: fadeIn 0.3s forwards;
}

/* User */
.rx-right { justify-content:flex-end; text-align:right; }
.rx-right .rx-bubble {
    color:#fff;
    background: linear-gradient(135deg, var(--rx-indigo) 0%, var(--rx-sky) 100%);
    box-shadow:0 5px 15px rgba(79,70,229,0.3);
    border-radius:20px 20px 4px 20px;
}

/* Assistant */
.rx-left { justify-content:flex-start; text-align:left; }
.rx-left .rx-bubble {
    color: var(--rx-text);
    background: rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 2px 10px rgba(0,0,0,0.15);
    border-radius:20px 20px 20px 4px;
}

/* Input bar */
form[data-testid="stForm"] {
    background: rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.12);
    border-radius:24px;
    padding:10px;
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
    position:fixed;
    bottom:10px;
    width:100%;
    max-width:900px;
    transform:translateX(-50%);
    left:50%;
    z-index:1000;
}

/* Input fields */
.stTextInput input, .stTextArea textarea {
    border-radius:20px !important;
    border:none !important;
    background: rgba(255,255,255,0.12) !important;
    color: var(--rx-text) !important;
    padding:15px 20px;
    min-height:50px;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    outline:2px solid var(--rx-indigo) !important;
}

/* Buttons */
.stButton>button {
    border-radius:16px;
    border:none;
    background: linear-gradient(90deg, var(--rx-indigo), var(--rx-indigo-600)) !important;
    color:#fff !important;
    font-weight:700;
    letter-spacing:0.02em;
    height:40px;
    padding:0 18px;
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 8px 20px rgba(79,70,229,0.5); }

/* Fade-in animation */
@keyframes fadeIn { to { opacity:1; } }
</style>
""", unsafe_allow_html=True)

# --- WELCOME MESSAGE ---
if len(st.session_state.messages) == 1:
    st.markdown(f"""
    <div class="welcome-card">
        <div>{st.session_state.messages[0]["content"]}</div>
    </div>
    """, unsafe_allow_html=True)

# --- DOCUMENT CHIPS ---
if st.session_state.sources:
    chips = "".join(f"<span class='rx-chip'>{s}</span>" for s in st.session_state.sources)
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)


# --- LOGIN / SIGNUP SCREEN ---
if not st.session_state.user:
    st.markdown(
        """
        <style>
        .center-box {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
            flex-direction: column;
            text-align: center;
        }
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .subtitle {
            color: gray;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="center-box">
            <div class="title">Welcome to RecallX 🔐</div>
            <div class="subtitle">Sign in or create an account to continue</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Sign Up"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "1234":  # example logic
                st.session_state.user = username
                st.success("Login successful! Reloading...")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_username = st.text_input("Choose a username")
        new_password = st.text_input("Choose a password", type="password")
        if st.button("Create Account", use_container_width=True):
            st.success("Account created! Please log in.")
    st.stop()
# --- ENV VARS & SETTINGS ---
load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


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




def make_http_client_no_proxy():
  
    return httpx.Client(timeout=60, trust_env=False)


def llm_answer(provider: str, model: str, api_key: str, question: str, context: str) -> str:
   
    system_msg = (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the context does not fully cover the answer, you may infer the most likely answer "
        "based on the context. Be concise and clear."
    )
    user_msg = f"Question: {question}\n\nContext:\n{context}\n"

    if provider == "groq":
        client = GroqClient(api_key=api_key, http_client=make_http_client_no_proxy())
        resp = client.chat.completions.create(
            model=model or "llama-3.1-8b-instant",
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":user_msg}],
            temperature=0.2,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    elif provider == "openai":
        client = OpenAIClient(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":user_msg}],
            temperature=0.2,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    elif provider == "deepseek":
        client = OpenAIClient(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":user_msg}],
            temperature=0.2,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    else:
        raise RuntimeError(f"Unknown provider: {provider}")



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
