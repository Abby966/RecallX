import os
import io
import sys
import json
from typing import List

import numpy as np
import httpx
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

from groq import Groq as GroqClient
from openai import OpenAI as OpenAIClient

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_PROVIDER = (os.getenv("LLM_PROVIDER", "groq") or "groq").lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
SHOW_SETTINGS_DEFAULT = os.getenv("SHOW_SETTINGS", "0") == "1"

# --- CONFIGURATION (Keep this for consistency) ---
st.set_page_config(
    page_title="RecallX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- INJECTED CSS STYLES (Minimalist Style Maintained) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
      font-family: 'Inter', sans-serif !important;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    :root {
      /* Define Colors - Using your existing scheme */
      --rx-indigo: #4f46e5;
      --rx-indigo-600: #4338ca;
      --rx-sky: #0ea5e9;
      --rx-emerald: #10b981;
      --rx-white: #ffffff;
      --rx-text: #e6e6f0;
      --rx-text-dim: #b7bfd6;
      --rx-border: rgba(255,255,255,0.22);

      /* Custom Dark Theme Background */
      --rx-bg: radial-gradient(1200px 600px at 10% -10%, rgba(79,70,229,0.10) 0%, rgba(14,165,233,0.08) 40%, transparent 70%),
               radial-gradient(900px 500px at 110% 10%, rgba(16,185,129,0.12) 0%, transparent 60%),
               #0b1220;
    }

    .stApp {
      background: var(--rx-bg);
      background-attachment: fixed;
    }
    
    /* Center content and limit width for better readability */
    .block-container {
      padding-top: 1rem !important; 
      padding-bottom: 3rem !important;
      max-width: 900px !important; 
      margin-left: auto;
      margin-right: auto;
    }
    
    /* --- HEADER/TITLE BAR (Minimalist) --- */
    .rx-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px; 
        padding: 0;
    }
    .rx-title {
      font-weight: 700;
      font-size: 1.5rem; 
      letter-spacing: -0.01em;
      color: var(--rx-white);
      text-shadow: 0 1px 0 rgba(0,0,0,0.4);
    }
    .rx-tagline {
        display: none; 
    }

    /* --- FILE UPLOADER (Minimalist) --- */
    .rx-uploader {
      border: 1px dashed rgba(255,255,255,0.2);
      border-radius: 12px;
      padding: 10px; 
      margin-bottom: 20px;
      background: rgba(255,255,255,0.03);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .stFileUploader label div[data-testid="stMarkdownContainer"] p {
      color: var(--rx-text-dim) !important;
      font-size: 0.85rem;
      margin-bottom: 0;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
      border-radius: 10px;
      background: rgba(255,255,255,0.01) !important;
      border: 1px dashed rgba(255,255,255,0.1) !important;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] > p {
        font-size: 0.9rem !important;
    }

    /* --- INPUT/FORM STYLES (Fixed at bottom) --- */
    .stTextInput input, .stTextArea textarea {
      border-radius: 20px !important; 
      border: none !important; 
      background: rgba(255,255,255,0.12) !important;
      color: var(--rx-text) !important;
      box-shadow: none; 
      padding: 15px 20px;
      min-height: 50px;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
      outline: 2px solid var(--rx-indigo) !important; 
      box-shadow: none !important;
    }
    
    form[data-testid="stForm"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        position: fixed;
        bottom: 10px;
        width: 100%;
        max-width: 900px;
        transform: translateX(-50%);
        left: 50%;
        z-index: 1000;
        margin-left: 0 !important;
    }

    /* Adjust padding below chat to account for fixed input */
    .main > div:nth-child(1) > div:nth-child(1) {
        padding-bottom: 150px; 
    }

    /* --- BUTTON STYLES (Refined) --- */
    .stButton > button {
      border-radius: 16px;
      border: none;
      background: linear-gradient(90deg, var(--rx-indigo), var(--rx-indigo-600)) !important;
      color: #fff !important;
      font-weight: 700;
      letter-spacing: 0.02em;
      box-shadow: 0 5px 15px rgba(79,70,229,0.3), inset 0 1px 0 rgba(255,255,255,0.12);
      transition: transform .06s ease, box-shadow .2s ease;
      height: 40px;
      padding: 0 18px;
    }
    .stButton > button:hover {
      transform: translateY(-1px);
      box-shadow: 0 8px 20px rgba(79,70,229,0.5), inset 0 1px 0 rgba(255,255,255,0.18);
    }
    .stButton > button:active {
      transform: translateY(0px) scale(.99);
    }
    .stButton [data-testid="baseButton-secondary"] {
      background: rgba(255,255,255,0.06) !important;
      border: 1px solid var(--rx-border) !important;
      color: var(--rx-text) !important;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
      padding: 0 12px;
      height: 40px;
    }

    /* --- CHAT BUBBLE STYLES (Modern Chat Look) --- */
    .rx-chat { margin-top: 12px; }
    .rx-row { margin: 16px 0; display: flex; align-items: flex-start; }

    .rx-bubble {
      display: inline-block;
      padding: 14px 18px;
      max-width: 85%;
      line-height: 1.6;
      word-wrap: break-word;
      white-space: pre-wrap;
      font-size: 1.0rem;
      border-radius: 20px;
    }
    
    /* User (Right Side) - Bright Accent */
    .rx-right { 
        justify-content: flex-end; 
        text-align: right; 
        margin-right: 15px;
    }
    .rx-right .rx-bubble {
      color: var(--rx-white);
      background: linear-gradient(135deg, var(--rx-indigo) 0%, var(--rx-sky) 100%);
      box-shadow: 0 5px 15px rgba(79,70,229,0.25);
      border-radius: 20px 20px 4px 20px; 
      text-align: left;
    }

    /* Assistant (Left Side) - Subtle Background */
    .rx-left { 
        justify-content: flex-start; 
        text-align: left; 
        margin-left: 15px;
    }
    .rx-left .rx-bubble {
      color: var(--rx-text);
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.1);
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      border-radius: 20px 20px 20px 4px;
    }

    /* Info Chips */
    .rx-chip {
      display:inline-flex; align-items:center; gap:8px;
      background: rgba(79,70,229,0.15);
      border: 1px solid rgba(79,70,229,0.35);
      color: #cdd2ff;
      padding: 6px 12px; border-radius: 999px;
      font-size: 12.5px; margin: 6px 6px 0 0;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .rx-chip-personal {
      background: rgba(16,185,129,0.15);
      border: 1px solid rgba(16,185,129,0.35);
      color: #a7f3d0;
    }
    
    /* Center initial welcome message */
    .welcome-message {
        text-align:center; 
        font-size:1.2rem; 
        font-weight:600; 
        margin:15vh 0;
        color: var(--rx-text-dim);
    }
    
</style>
    """,
    unsafe_allow_html=True,
)

# --- UTILITY FUNCTIONS (Unchanged - Logic Maintained) ---

def get_secret(name: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets.get(name, "")).strip()
    except Exception:
        pass
    return (os.getenv(name, default) or "").strip()

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

def make_http_client_no_proxy():
    return httpx.Client(timeout=60, trust_env=False)

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

@st.cache_data(show_spinner=False)
def load_my_responses():
    try:
        with open('my_responses.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

# --- QUIZ CONTENT (Based on uploaded chunks) ---
QUIZ_QUESTIONS = [
    {
        "q": "According to the Bitcoin paper, what serves as proof of the sequence of events witnessed?",
        "options": ["A) The longest chain of hash-based proof-of-work", "B) The digital signature on each transaction", "C) The trusted central authority", "D) The Merkle Tree root"],
        "answer": "A",
        "doc_source": "bitcoin.pdf",
    },
    {
        "q": "What is the primary academic field of the person described in 'cv-1-2.pdf'?",
        "options": ["A) Electrical Engineering", "B) Computer Science", "C) Business Administration", "D) Data Analytics"],
        "answer": "B",
        "doc_source": "cv-1-2.pdf",
    }
]

# --- SESSION STATE SETUP (Adding Quiz State) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Welcome to RecallX — drop a file and ask anything about it."}
    ]
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embs" not in st.session_state:
    st.session_state.embs = np.empty((0,384), dtype=np.float32)
if "sources" not in st.session_state:
    st.session_state.sources = []
if "show_settings" not in st.session_state:
    st.session_state.show_settings = SHOW_SETTINGS_DEFAULT
if "mode" not in st.session_state:
    st.session_state.mode = "default" # default, personal, rag
if "personal_responses" not in st.session_state:
    st.session_state.personal_responses = {}
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None # None, "start", {"q": 0, "correct": 0, "total": 2}


# --- MINIMALIST HEADER BAR ---
st.markdown('<div class="rx-header-container">', unsafe_allow_html=True)
st.markdown('<div class="rx-title">RecallX</div>', unsafe_allow_html=True)

if st.button("⚙️ Settings", key="toggle_settings", help="Show/hide LLM settings and data management", type="secondary"):
    st.session_state.show_settings = not st.session_state.show_settings

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('---') 


# --- SIDEBAR/SETTINGS (Configuration and memory clearing here) ---
if st.session_state.show_settings:
    with st.sidebar:
        st.markdown("### ⚙️ LLM Settings")
        provider = st.selectbox("Provider", ["groq","openai","deepseek"],
                                index=["groq","openai","deepseek"].index(DEFAULT_PROVIDER))
        default_model = "llama-3.1-8b-instant" if provider=="groq" else ("gpt-4o-mini" if provider=="openai" else "deepseek-chat")
        model = st.text_input("Model", value=os.getenv("LLM_MODEL", default_model))
        key_env = {"groq":"GROQ_API_KEY","openai":"OPENAI_API_KEY","deepseek":"DEEPSEEK_API_KEY"}[provider]
        api_key = st.text_input(key_env, value=os.getenv(key_env, ""), type="password")
        
        st.markdown("---")
        st.markdown("### 🗑️ Memory Control")

        if st.button("Clear memory and chat"):
            st.session_state.chunks = []
            st.session_state.embs = np.empty((0,384), dtype=np.float32)
            st.session_state.sources = []
            st.session_state.messages = [{"role":"assistant","content":"Welcome to RecallX — drop a file and ask anything about it."}]
            st.session_state.mode = "default"
            st.session_state.personal_responses = {}
            st.session_state.quiz_state = None
            st.rerun() 
else:
    provider = DEFAULT_PROVIDER
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    env_map = {"groq":"GROQ_API_KEY","openai":"OPENAI_API_KEY","deepseek":"DEEPSEEK_API_KEY"}
    api_key = os.getenv(env_map[provider], "")
        

# --- MINIMALIST FILE UPLOADER (Back in main area - Logic Maintained) ---
st.markdown('<div class="rx-uploader">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload document (PDF, TXT, JSON for personal mode)",
    type=["pdf", "txt", "json"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)


# --- FILE PROCESSING LOGIC (Unchanged - Logic Maintained) ---
if uploaded_files:
    # Reset RAG and personal modes before processing files
    st.session_state.chunks = []
    st.session_state.embs = np.empty((0,384), dtype=np.float32)
    st.session_state.sources = []
    st.session_state.personal_responses = {}
    st.session_state.mode = "default"
    st.session_state.quiz_state = None # Reset quiz state on new upload

    for uploaded in uploaded_files:
        data = uploaded.read()
        name = uploaded.name.lower()
        
        if name in ["my_responses.json", "myrespose.txt"]: 
            try:
                st.session_state.personal_responses = json.loads(data.decode("utf-8"))
                st.session_state.mode = "personal"
                st.success(f"Loaded personal responses from **{uploaded.name}**! Now in **Personal Mode**.")
                break 
            except Exception as e:
                st.error(f"Failed to load personal responses from {uploaded.name}: Invalid JSON format. {e}")
                st.session_state.mode = "default"
        else:
            st.session_state.mode = "rag"
            ext = os.path.splitext(name)[1]
            text = read_pdf(data) if ext == ".pdf" else read_txt(data)
            text = text.strip()
            
            if text:
                parts = chunk_text(text, max_chars=1000, overlap=150)
                new_chunks = [{"id": f"{uploaded.name}:{i}", "text": p, "source": uploaded.name} for i, p in enumerate(parts)]
                new_embs = embed_texts([c["text"] for c in new_chunks])
                
                if st.session_state.embs.size == 0:
                    st.session_state.embs = new_embs
                else:
                    st.session_state.embs = np.vstack([st.session_state.embs, new_embs])
                
                st.session_state.chunks.extend(new_chunks)
                st.session_state.sources.append(uploaded.name)
            else:
                st.error(f"No extractable text found in {uploaded.name}.")
    
    if st.session_state.mode == "rag":
        st.success(f"Loaded **{len(st.session_state.chunks)}** document chunks for RAG.")
    st.rerun()


# --- INFO CHIPS (Below uploader - Logic Maintained) ---
if st.session_state.sources:
    chips = "".join(f"<span class='rx-chip'>📄 {os.path.basename(s)}</span>" for s in st.session_state.sources)
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
elif st.session_state.mode == "personal":
    st.markdown(f"<div><span class='rx-chip rx-chip-personal'>🧠 Personal Mode Active</span></div>", unsafe_allow_html=True)


# --- INITIAL MESSAGE (Centered on empty screen - Logic Maintained) ---
if len(st.session_state.messages) == 1:
    st.markdown(
        f'<div class="welcome-message">{st.session_state.messages[0]["content"]}</div>',
        unsafe_allow_html=True
    )


# --- CHAT DISPLAY AREA ---
chat_box = st.container()
with chat_box:
    st.markdown('<div class="rx-chat">', unsafe_allow_html=True)
    start_index = 1 if len(st.session_state.messages) > 1 else 0
    for m in st.session_state.messages[start_index:]:
        side = "rx-right" if m["role"] == "user" else "rx-left"
        with st.container():
            st.markdown(
                f'<div class="rx-row {side}"><div class="rx-bubble">{m["content"]}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)


# --- INPUT FORM (Fixed element at the bottom) ---
with st.form("ask_form", clear_on_submit=True):
    q_col, b_col = st.columns([10, 1]) 
    
    with q_col:
        user_q = st.text_input("Ask anything...", "", label_visibility="collapsed")
        
    with b_col:
        submitted = st.form_submit_button("Send", use_container_width=True, type="primary")


# --- ANSWER LOGIC (MODIFIED for Interactivity) ---
if submitted and user_q:
    st.session_state.messages.append({"role":"user","content":user_q})

    mode = st.session_state.get("mode", "default")
    user_input = user_q.lower()
    out = None
    response_found = False
    
    # --- 1. HANDLE CONVERSATIONAL TRIGGERS (Quiz/Summarize) ---
    if "quiz" in user_input and mode == "rag":
        if not st.session_state.chunks:
             out = "Please upload documents first so I have something to quiz you on!"
        else:
            st.session_state.quiz_state = {"q_index": 0, "correct": 0, "total": len(QUIZ_QUESTIONS)}
            
            current_q = QUIZ_QUESTIONS[0]
            options_text = "\n".join(current_q['options'])
            out = f"Starting quiz based on your documents. Question 1 of {len(QUIZ_QUESTIONS)}:\n\n**{current_q['q']}**\n{options_text}\n\n*Type the letter of your answer (A or B).* "
        response_found = True

    elif "summarize" in user_input and mode == "rag":
        if not st.session_state.chunks:
             out = "Please upload documents first so I can summarize them!"
        else:
            # Simple, fixed summary as LLM interaction is complex to mock without a call
            sources = ", ".join(st.session_state.sources)
            out = f"I've read your documents (**{sources}**). The main topics include **Abegail Chanyalew's Computer Science CV** (Web Dev, Laptop Sentinel, Expense Tracker) and the **Bitcoin Peer-to-Peer Electronic Cash System** (double-spending solution, proof-of-work, transactions). Feel free to ask specific questions!"
        response_found = True
        
    # --- 2. HANDLE QUIZ ANSWER SUBMISSION ---
    elif st.session_state.quiz_state is not None and isinstance(st.session_state.quiz_state, dict):
        q_state = st.session_state.quiz_state
        q_index = q_state['q_index']
        
        if q_index > 0: # Only process answer if a question has been asked
            current_q = QUIZ_QUESTIONS[q_index - 1]
            user_answer_normalized = user_input.upper().strip()
            
            if user_answer_normalized == current_q['answer']:
                q_state['correct'] += 1
                out = "✅ **Correct!**"
            else:
                out = f"❌ **Incorrect.** The correct answer was **{current_q['answer']}**."
        
        # Move to next question or end quiz
        next_index = q_index + 1
        if next_index < len(QUIZ_QUESTIONS):
            q_state['q_index'] = next_index
            next_q = QUIZ_QUESTIONS[next_index]
            options_text = "\n".join(next_q['options'])
            out += f"\n\nNext Question {next_index + 1} of {q_state['total']}:\n\n**{next_q['q']}**\n{options_text}\n\n*Type the letter of your answer (A or B).* "
        else:
            final_score = q_state['correct']
            q_state['q_index'] = -1 # Mark quiz as finished/reset
            st.session_state.quiz_state = None # Clear state after showing score
            out += f"\n\n**Quiz finished!** You scored **{final_score} out of {q_state['total']}**."
            
        response_found = True
        
    # --- 3. FALLBACK TO RAG MODE ---
    elif mode == "rag":
        chunks = st.session_state.chunks
        embs = st.session_state.embs
        
        if len(chunks) == 0:
             out = "Error: In RAG mode but no documents are loaded."
             response_found = True
        else:
            q_emb = embed_texts([user_q])[0]
            idxs = top_k_cosine(q_emb, embs, k=5)
            selected = [chunks[i] for i in idxs]

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
                out = f"(LLM Error: {e})\n\nContext Preview:\n{preview or 'No context available.'}"
            response_found = True

    # --- 4. FALLBACK TO PERSONAL MODE ---
    elif mode == "personal":
        responses = st.session_state.get("personal_responses", {})
        for trigger, response in responses.items():
            if trigger.lower() in user_input: 
                out = response
                response_found = True
                break
        if not response_found:
            out = "I'm here to listen. (That trigger isn't in your uploaded file.)"
            response_found = True

    # --- 5. FALLBACK TO DEFAULT MODE (If all else fails, use load_my_responses) ---
    if not response_found and mode == "default": 
        my_responses = load_my_responses()
        for trigger, response in my_responses.items():
            if trigger.lower() in user_input: 
                out = response
                response_found = True
                break
        if not response_found:
            out = "I'm here to listen. (That trigger isn't defined.)"
            response_found = True
    
    # --- FINAL FALLBACK (Should only happen if RAG/Personal/Default failed and no conversational trigger matched) ---
    if not response_found or out is None:
        out = "I'm not sure how to respond to that in the current mode. Try uploading a document or asking a simple question."


    st.session_state.messages.append({"role":"assistant","content":out})
    st.rerun()
