import os
import io
import sys
from typing import List

import numpy as np
import httpx
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

# Providers
from groq import Groq as GroqClient
from openai import OpenAI as OpenAIClient  # also used for DeepSeek via base_url

# --------- ENV / Defaults ---------
load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PROVIDER = (os.getenv("LLM_PROVIDER", "groq") or "groq").lower()  # groq | openai | deepseek
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
SHOW_SETTINGS_DEFAULT = os.getenv("SHOW_SETTINGS", "0") == "1"  # sidebar hidden by default

# --------- Page config ---------
st.set_page_config(
    page_title="RecallX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------- Light theme (white + blue) + chat bubbles ---------
st.markdown(
    """
    <style>
  /* Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol", sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* Color tokens */
  :root {
    --rx-indigo: #4f46e5;
    --rx-indigo-600: #4338ca;
    --rx-sky: #0ea5e9;
    --rx-emerald: #10b981;

    --rx-bg: radial-gradient(1200px 600px at 10% -10%, rgba(79,70,229,0.10) 0%, rgba(14,165,233,0.08) 40%, transparent 70%),
             radial-gradient(900px 500px at 110% 10%, rgba(16,185,129,0.12) 0%, transparent 60%),
             #0b1220;
    --rx-surface: rgba(255,255,255,0.06);
    --rx-card: rgba(255,255,255,0.10);
    --rx-border: rgba(255,255,255,0.22);
    --rx-text: #e6e6f0;
    --rx-text-dim: #b7bfd6;
    --rx-white: #ffffff;
  }

  .stApp {
    background: var(--rx-bg);
    background-attachment: fixed;
  }

  /* Global container width */
  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
  }

  /* Glass cards */
  .rx-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid var(--rx-border);
    border-radius: 18px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 18px 18px 14px 18px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
  }

  /* Header title */
  .rx-title {
    font-weight: 800;
    font-size: 2.3rem;
    letter-spacing: -0.02em;
    color: var(--rx-white);
    margin-bottom: 4px;
    text-shadow: 0 1px 0 rgba(0,0,0,0.4);
  }
  .rx-tagline {
    color: var(--rx-text-dim);
    font-weight: 500;
  }

  /* Buttons */
  .stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.18);
    background: linear-gradient(90deg, var(--rx-indigo), var(--rx-indigo-600)) !important;
    color: #fff !important;
    font-weight: 700;
    letter-spacing: 0.02em;
    box-shadow: 0 8px 24px rgba(79,70,229,0.35), inset 0 1px 0 rgba(255,255,255,0.12);
    transition: transform .06s ease, box-shadow .2s ease, filter .2s ease;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 26px rgba(79,70,229,0.45), inset 0 1px 0 rgba(255,255,255,0.18);
    filter: saturate(1.05);
  }
  .stButton > button:active {
    transform: translateY(0px) scale(.99);
  }

  /* Secondary button */
  .stButton [data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--rx-border) !important;
    color: var(--rx-text) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
  }

  /* Inputs */
  .stTextInput input, .stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid var(--rx-border) !important;
    background: rgba(255,255,255,0.06) !important;
   color: #000000 !important;   /* black text */
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(79,70,229,0.65) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.25) !important;
  }

  /* File uploader */
  .rx-uploader {
    border: 1px dashed rgba(255,255,255,0.28);
    border-radius: 16px;
    padding: 18px;
    background: rgba(255,255,255,0.05);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
  }
  .stFileUploader label div[data-testid="stMarkdownContainer"] p {
    color: var(--rx-text-dim) !important;
  }
  .stFileUploader div[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
    border: 1px dashed rgba(255,255,255,0.28) !important;
  }

  /* Chips */
  .rx-chip {
    display:inline-flex; align-items:center; gap:8px;
    background: rgba(79,70,229,0.15);
    border: 1px solid rgba(79,70,229,0.35);
    color: #cdd2ff;
    padding: 6px 10px; border-radius: 999px;
    font-size: 12.5px; margin: 6px 6px 0 0;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
  }

  /* Chat area */
  .rx-chat { margin-top: 12px; }
  .rx-row { margin: 10px 0; }

  /* Bubbles */
  .rx-bubble {
    display: inline-block;
    padding: 12px 14px;
    border-radius: 18px;
    max-width: 78%;
    background: rgba(255,255,255,0.06);
    color: var(--rx-text);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 10px 26px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.06);
    line-height: 1.5;
    word-wrap: break-word;
    white-space: pre-wrap;
    font-size: 0.975rem;
    transition: transform .06s ease;
  }
  .rx-left  { text-align: left; }
  .rx-left  .rx-bubble {
    border-radius: 18px 18px 18px 6px;
    background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
  }

  .rx-right { text-align: right; }
  .rx-right .rx-bubble {
    color: #ffffff;
    background: linear-gradient(135deg, var(--rx-indigo) 0%, var(--rx-sky) 100%);
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 14px 30px rgba(79,70,229,0.40), inset 0 1px 0 rgba(255,255,255,0.22);
    border-radius: 18px 18px 6px 18px;
  }
  .rx-bubble:hover { transform: translateY(-1px); }

  /* Form submit alignment tweaks */
  form[data-testid="stForm"] .stFormSubmitButton {
    padding-top: 0.25rem;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 12px; height: 12px; }
  ::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(79,70,229,0.45), rgba(14,165,233,0.45));
    border: 3px solid rgba(0,0,0,0);
    background-clip: padding-box;
    border-radius: 8px;
  }
  ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }

  /* Footer watermark */
  .rx-footer {
    color: var(--rx-text-dim);
    font-size: 12.75px;
    text-align: center;
    margin-top: 28px;
    opacity: .9;
  }
</style>

    """,
    unsafe_allow_html=True,
)

# --------- Secrets/helper ---------
def get_secret(name: str, default: str = "") -> str:
    """Priority: Streamlit Secrets -> OS env -> default"""
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets.get(name, "")).strip()
    except Exception:
        pass
    return (os.getenv(name, default) or "").strip()

# --------- Embeddings / utils ---------
@st.cache_resource(show_spinner=False)
def get_model(name: str):
    return SentenceTransformer(name, trust_remote_code=True)

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model(EMBED_MODEL)
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embs.astype(np.float32)

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

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

# --------- No-proxy HTTP client ---------
def make_http_client_no_proxy():
    return httpx.Client(timeout=60, trust_env=False)
# --------- LLM answer ---------
def llm_answer(provider: str, model: str, api_key: str, question: str, context: str) -> str:
    system_msg = (
        "You are a precise assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say: 'I couldn’t find this in the document.' "
        "Be concise and clear. Include key facts, not speculation."
    )
    user_msg = f"Question: {question}\n\nContext:\n{context}\n"

    if provider == "groq":
        client = GroqClient(api_key=api_key, http_client=make_http_client_no_proxy())
        resp = client.chat.completions.create(
            model=model or "llama-3.1-8b-instant",
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=0.2, max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    if provider == "openai":
        client = OpenAIClient(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=0.2, max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    if provider == "deepseek":
        client = OpenAIClient(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=0.2, max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    raise RuntimeError(f"Unknown provider: {provider}")

# --------- Session state ---------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Welcome to RecallX — drop a file and ask anything about it."}
    ]
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embs" not in st.session_state:
    st.session_state.embs = np.empty((0,384), dtype=np.float32)
if "sources" not in st.session_state:
    st.session_state.sources = []  # list of filenames
if "show_settings" not in st.session_state:
    st.session_state.show_settings = SHOW_SETTINGS_DEFAULT  # hidden by default

# --------- Header + Settings toggle ---------
st.markdown('<div class="rx-card">', unsafe_allow_html=True)
c1, c2 = st.columns([1,1])
with c1:
    st.markdown('<div class="rx-title">RecallX</div>', unsafe_allow_html=True)
    st.markdown('<div class="rx-tagline">Upload a document • Get answers • Stay in flow</div>', unsafe_allow_html=True)
with c2:
    col_a, col_b = st.columns([0.7,0.3])
    with col_b:
        if st.button("⚙️ Settings", key="toggle_settings", help="Show/hide provider & API key", type="secondary"):
            st.session_state.show_settings = not st.session_state.show_settings
st.markdown('</div>', unsafe_allow_html=True)

# --------- Settings (hidden by default; toggle to show) ---------
if st.session_state.show_settings:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        provider = st.selectbox("Provider", ["groq","openai","deepseek"],
                                index=["groq","openai","deepseek"].index(DEFAULT_PROVIDER))
        default_model = "llama-3.1-8b-instant" if provider=="groq" else ("gpt-4o-mini" if provider=="openai" else "deepseek-chat")
        model = st.text_input("Model", value=os.getenv("LLM_MODEL", default_model))
        key_env = {"groq":"GROQ_API_KEY","openai":"OPENAI_API_KEY","deepseek":"DEEPSEEK_API_KEY"}[provider]
        api_key = st.text_input(key_env, value=os.getenv(key_env, ""), type="password")
        if st.button("Clear memory"):
            st.session_state.chunks = []
            st.session_state.embs = np.empty((0,384), dtype=np.float32)
            st.session_state.sources = []
            st.session_state.messages = [{"role":"assistant","content":"Welcome to RecallX — drop a file and ask anything about it."}]
else:
    provider = DEFAULT_PROVIDER
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    env_map = {"groq":"GROQ_API_KEY","openai":"OPENAI_API_KEY","deepseek":"DEEPSEEK_API_KEY"}
    api_key = os.getenv(env_map[provider], "")

# --------- Uploader (MULTI-FILE) ---------
st.markdown('<div class="rx-card">', unsafe_allow_html=True)
st.markdown('<div class="rx-uploader">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload one or more PDFs or text files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle multi-file upload
if uploaded_files:
    for uploaded in uploaded_files:
        data = uploaded.read()
        name = uploaded.name
        ext = os.path.splitext(name)[1].lower()
        text = read_pdf(data) if ext == ".pdf" else read_txt(data)
        text = text.strip()
        if text:
            parts = chunk_text(text, max_chars=1000, overlap=150)
            new_chunks = [{"id": f"{name}:{i}", "text": p, "source": name} for i, p in enumerate(parts)]
            new_embs = embed_texts([c["text"] for c in new_chunks])
            if st.session_state.embs.size == 0:
                st.session_state.embs = new_embs
            else:
                st.session_state.embs = np.vstack([st.session_state.embs, new_embs])
            st.session_state.chunks.extend(new_chunks)
            st.session_state.sources.append(name)
            st.success(f"Loaded {name} with {len(new_chunks)} chunks.")
        else:
            st.error(f"No extractable text found in {name}.")

# Show file chips if any
if st.session_state.sources:
    chips = "".join(f"<span class='rx-chip'>📄 {os.path.basename(s)}</span>" for s in st.session_state.sources)
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

# --------- Welcome centered (only at start) ---------
if len(st.session_state.messages) == 1:
    st.markdown(
        "<div style='text-align:center; font-size:1.2rem; font-weight:600; margin:16px 0;'>"
        + st.session_state.messages[0]["content"] +
        "</div>",
        unsafe_allow_html=True
    )

# --------- Ask (TOP) ---------
with st.form("ask_form", clear_on_submit=True):
    user_q = st.text_input("Ask anything about your document…", "")
    submitted = st.form_submit_button("Ask", use_container_width=True)
if submitted and user_q:
    st.session_state.messages.append({"role":"user","content":user_q})

    # Retrieval + LLM
    chunks = st.session_state.chunks
    embs = st.session_state.embs
    if len(chunks) == 0:
        out = "No documents uploaded yet. Please upload a PDF or .txt first."
    else:
        q_emb = embed_texts([user_q])[0]
        idxs = top_k_cosine(q_emb, embs, k=5)
        selected = [chunks[i] for i in idxs]

        # bounded context ~6k chars
        pieces, total = [], 0
        for s in selected:
            t = f"[{s['source']}] {s['text']}"
            if total + len(t) > 6000: break
            pieces.append(t); total += len(t)
        context = "\n\n".join(pieces)

        try:
            if not api_key:
                raise RuntimeError("Missing API key. (Open settings or set env vars.)")
            out = llm_answer(provider, model, api_key, user_q, context)
        except Exception as e:
            preview = "\n".join(context.splitlines()[:8])
            out = f"(Fallback: {e})\n\n{preview or 'No context available.'}"

    st.session_state.messages.append({"role":"assistant","content":out})
    st.rerun()
    # refresh to render new bubbles below

# --------- Chat timeline (BOTTOM) ---------
chat_box = st.container()
with chat_box:
    st.markdown('<div class="rx-chat">', unsafe_allow_html=True)
    # skip the first welcome message (already centered)
    for m in st.session_state.messages[1:]:
        side = "rx-right" if m["role"] == "user" else "rx-left"
        safe = m["content"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="rx-row {side}"><div class="rx-bubble">{safe}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)
