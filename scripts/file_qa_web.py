#!/usr/bin/env python3
"""
Web-based File Q&A Interface - ChatGPT-style
Modern chat interface for document Q&A using Streamlit
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.file_qa import FileQA
from src.auth.firebase import FirebaseUser
from src.auth.google_oauth import (
    build_session_token,
    build_authorize_url,
    exchange_code_for_tokens,
    fetch_userinfo,
    load_google_oauth_config,
    verify_google_id_token,
    verify_session_token,
    verify_state,
)
from src.storage.firestore_store import FirestoreStore
from src.storage.gcs_store import GCSStore
from src.privacy.redaction import redact_sources, redact_text, sanitize_filename
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Lab Lens - File Q&A",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",  # Ensure sidebar is expanded by default
)

# Custom CSS for ChatGPT-like dark interface
st.markdown(
    """
<style>
  /* Hide Streamlit branding but keep sidebar toggle */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  /* Don't hide header - it contains the sidebar toggle button */
  /* header {visibility: hidden;} */
 
  /* Main container - full height layout */
  .main {
    padding-top: 0rem;
    padding-bottom: 0rem;
  }
 
  /* Block container - ensure proper spacing */
  .block-container {
    padding-bottom: 100px !important; /* Space for fixed input */
  }
 
  /* Fixed chat input at bottom - always visible */
  .chat-input-container {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    background-color: #0e1117 !important;
    padding: 1rem !important;
    border-top: 1px solid #343541 !important;
    z-index: 999 !important;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.3) !important;
  }
 
  /* Adjust for sidebar */
  [data-testid="stSidebar"][aria-expanded="true"] ~ .main .chat-input-container {
    left: 21rem !important;
  }
 
  @media (max-width: 768px) {
    [data-testid="stSidebar"] ~ .main .chat-input-container {
      left: 0 !important;
    }
  }
 
  /* Ensure page is scrollable */
  html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
    overflow-y: auto;
  }
 
  /* Messages area - allow natural scrolling */
  .element-container {
    margin-bottom: 1rem;
  }
 
  /* Scroll to bottom button (appears when user scrolls up) */
  .scroll-to-bottom-btn {
    position: fixed;
    bottom: 100px;
    right: 2rem;
    background-color: #40414f;
    border: 1px solid #565869;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    display: none;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 998;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease;
  }
 
  .scroll-to-bottom-btn:hover {
    background-color: #4a4b59;
    transform: scale(1.1);
  }
 
  .scroll-to-bottom-btn.visible {
    display: flex;
  }
 
  /* Adjust for sidebar */
  [data-testid="stSidebar"][aria-expanded="true"] ~ .main .scroll-to-bottom-btn {
    right: calc(2rem + 21rem);
  }
 
  /* Sidebar styling - ChatGPT-like - FORCE VISIBILITY */
  [data-testid="stSidebar"] {
    background-color: #171717 !important;
    padding: 1rem !important;
    min-width: 280px !important;
    max-width: 350px !important;
    visibility: visible !important;
    display: block !important;
    position: relative !important;
    z-index: 100 !important;
  }
 
  [data-testid="stSidebar"][aria-expanded="true"],
  [data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 280px !important;
    visibility: visible !important;
    display: block !important;
  }
 
  [data-testid="stSidebar"] * {
    color: #ececec !important;
  }
 
  /* Make sure sidebar content is visible */
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    visibility: visible !important;
    display: block !important;
  }
 
  /* Sidebar toggle button - make it visible and functional */
  button[data-testid="baseButton-header"] {
    visibility: visible !important;
    display: block !important;
  }
 
  /* Ensure main content adjusts for sidebar */
  .main {
    margin-left: 280px !important;
  }
 
  /* Sidebar header */
  .sidebar-header {
    padding: 1rem;
    border-bottom: 1px solid #343541;
  }
 
  /* Search bar in sidebar */
  .sidebar-search {
    width: 100%;
    padding: 0.5rem;
    background-color: #343541;
    border: 1px solid #565869;
    border-radius: 8px;
    color: #ececec;
    font-size: 14px;
  }
 
  .sidebar-search:focus {
    outline: none;
    border-color: #565869;
  }
 
  /* Chat history items */
  .chat-history-item {
    padding: 0.75rem 1rem;
    margin: 0.25rem 0.5rem;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.2s;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 14px;
  }
 
  .chat-history-item:hover {
    background-color: #343541;
  }
 
  .chat-history-item.active {
    background-color: #343541;
    border-left: 3px solid #10a37f;
  }
 
  /* New chat button */
  .new-chat-button {
    width: 100%;
    padding: 0.75rem;
    margin: 0.5rem;
    background-color: transparent;
    border: 1px solid #565869;
    border-radius: 8px;
    color: #ececec;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 14px;
  }
 
  .new-chat-button:hover {
    background-color: #343541;
    border-color: #10a37f;
  }
 
  /* User profile at bottom */
  .user-profile {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 1rem;
    border-top: 1px solid #343541;
    background-color: #171717;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
 
  .user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 14px;
  }
 
  /* Sidebar content area - scrollable */
  .sidebar-content {
    height: calc(100vh - 200px);
    overflow-y: auto;
    padding-bottom: 80px;
  }
 
  /* Hide default Streamlit sidebar elements we don't need */
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    padding: 0;
  }
 
  /* Chat input wrapper - makes columns look like one component */
  .chat-input-wrapper {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 0;
    max-width: 100%;
  }
 
  /* Remove gap between columns */
  .chat-input-wrapper [data-testid="column"] {
    padding: 0 !important;
  }
 
  /* Button styled to look like part of input */
  .chat-input-wrapper button[key="add_files_button"] {
    background-color: #40414f !important;
    border: 1px solid #565869 !important;
    border-right: none !important;
    border-radius: 12px 0 0 12px !important;
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 20px !important;
    padding: 0 !important;
    min-width: 48px !important;
    width: 100% !important;
    height: 56px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
  }
 
  .chat-input-wrapper button[key="add_files_button"]:hover {
    background-color: #4a4b59 !important;
    color: rgba(255, 255, 255, 1) !important;
  }
 
  /* Chat input styled to connect seamlessly */
  .chat-input-wrapper .stChatInput > div > div > div {
    background-color: #40414f !important;
    border-radius: 0 !important;
    border: 1px solid #565869 !important;
    border-left: none !important;
    border-right: 1px solid #565869 !important;
    margin: 0 !important;
  }
 
  /* Send button (if visible) */
  .chat-input-wrapper .stChatInput button {
    border-radius: 0 12px 12px 0 !important;
  }
 
  .chat-input-wrapper .stChatInput > div > div > div > textarea {
    color: white !important;
    font-size: 16px !important;
  }
 
  /* Button styling */
  .stButton > button {
    border-radius: 8px;
    border: 1px solid #565869;
    background-color: #343541;
    color: white;
    width: 100%;
  }
 
  .stButton > button:hover {
    background-color: #40414f;
  }
 
  /* File uploader styling */
  .uploadedFile {
    background-color: #343541;
    border-radius: 8px;
    padding: 0.5rem;
    margin: 0.5rem 0;
  }
 
  /* Welcome message */
  .welcome-container {
    text-align: center;
    padding: 4rem 2rem;
  }
 
  .welcome-title {
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
  }
 
  .welcome-subtitle {
    color: #888;
    font-size: 1rem;
  }
 
  /* Ensure messages are displayed in order */
  [data-testid="stChatMessage"] {
    margin-bottom: 1rem;
  }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_firestore_store() -> FirestoreStore:
    return FirestoreStore()


@st.cache_resource
def get_gcs_store() -> Optional[GCSStore]:
    """
    Returns a GCS store if configured; otherwise None (local dev without persistence).
    """
    try:
        return GCSStore()
    except Exception:
        return None


def firebase_web_config() -> Dict[str, str]:
    """
    Firebase client (browser) config for Google sign-in.

    Required env vars:
      - FIREBASE_API_KEY
      - FIREBASE_AUTH_DOMAIN
      - FIREBASE_PROJECT_ID
      - FIREBASE_APP_ID
    """
    cfg = {
        "apiKey": (os.getenv("FIREBASE_API_KEY") or "").strip(),
        "authDomain": (os.getenv("FIREBASE_AUTH_DOMAIN") or "").strip(),
        "projectId": (os.getenv("FIREBASE_PROJECT_ID") or "").strip(),
        "appId": (os.getenv("FIREBASE_APP_ID") or "").strip(),
    }
    return cfg


def _get_query_param(name: str) -> Optional[str]:
    """Query param helper compatible with old/new Streamlit APIs."""
    try:
        v = st.query_params.get(name)  # type: ignore[attr-defined]
        if v is None:
            return None
        if isinstance(v, list):
            return v[0] if v else None
        return str(v)
    except Exception:
        qp = st.experimental_get_query_params()
        arr = qp.get(name)
        return arr[0] if arr else None


def render_auth_session_bridge() -> None:
    """
    Bridge browser storage -> Streamlit session_state via a hidden text input.

    Streamlit sessions reset on full page refresh; this restores login state from localStorage.
    """
    st.text_input("auth_session_token", key="auth_session_token", label_visibility="collapsed")
    st.markdown(
        """
<style>
  div[data-testid="stTextInput"] label:has(+ div input[aria-label="auth_session_token"]) {display:none;}
  div[data-testid="stTextInput"] div:has(input[aria-label="auth_session_token"]) {display:none;}
</style>
""",
        unsafe_allow_html=True,
    )

    components.html(
        """
<script>
  (function () {
    function readToken() {
      try {
        if (window.top && window.top.localStorage) {
          return window.top.localStorage.getItem("lab_lens_auth_session") || "";
        }
      } catch (e) {}
      try {
        return localStorage.getItem("lab_lens_auth_session") || "";
      } catch (e) {}
      return "";
    }

    function setStreamlitWidgetValue(token) {
      try {
        const doc = window.parent && window.parent.document ? window.parent.document : document;
        const input = doc.querySelector('input[aria-label="auth_session_token"]');
        if (!input) return false;
        if ((input.value || "") === (token || "")) return true;
        input.value = token || "";
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      } catch (e) {
        return false;
      }
    }

    // Widget can be missing on first paint; retry briefly.
    const token = readToken();
    let tries = 0;
    const maxTries = 50; // ~10s at 200ms
    const timer = setInterval(function () {
      tries += 1;
      const ok = setStreamlitWidgetValue(token);
      if (ok || tries >= maxTries) clearInterval(timer);
    }, 200);
  })();
</script>
""",
        height=0,
    )


def render_google_sign_in() -> None:
    """
    Renders reliable Google Sign-in using OAuth redirect (server-side code exchange).

    This avoids Firebase JS popups, which often fail inside Streamlit's sandboxed iframes on Cloud Run.
    """
    cfg = load_google_oauth_config()
    if not cfg:
        st.error(
            "Google Sign-in is not configured. Set GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REDIRECT_URI."
        )
        return

    auth_url = build_authorize_url(cfg)
    # Streamlit version compatibility: link_button was added later than some 1.x releases.
    if hasattr(st, "link_button"):
        st.link_button("Sign in with Google", auth_url, use_container_width=True, type="primary")  # type: ignore[attr-defined]
    else:
        st.markdown(
            f"""
<a href="{auth_url}" target="_self" style="text-decoration:none;">
  <div style="width:100%; text-align:center; padding:0.6rem 0.8rem; border-radius:8px; border:1px solid #565869; background:#343541; color:#fff;">
    Sign in with Google
  </div>
</a>
""",
            unsafe_allow_html=True,
        )


def ensure_user() -> Optional[FirebaseUser]:
    """Return currently signed-in user (restored from localStorage-backed session token)."""
    user = st.session_state.get("user")
    if user:
        return user

    token = (st.session_state.get("auth_session_token") or "").strip()
    if not token:
        return None

    cfg = load_google_oauth_config()
    if not cfg:
        return None

    payload = verify_session_token(token, cfg.client_secret)
    if not payload:
        return None

    fb_user = FirebaseUser(
        uid=str(payload.get("uid") or ""),
        email=str(payload.get("email")) if payload.get("email") else None,
        name=str(payload.get("name")) if payload.get("name") else None,
        picture=str(payload.get("picture")) if payload.get("picture") else None,
    )
    if not fb_user.uid:
        return None

    st.session_state.user = fb_user
    return fb_user


def initialize_qa_system(
    user_id: Optional[str] = None,
    use_biobert: bool = False,
    *,
    privacy_mode: bool = True,
    allow_external_calls: bool = True,
    pii_extra_terms: Optional[list[str]] = None,
):
    """
    Initialize the File QA system with optional user-specific vector database

    Args:
      user_id: Optional user ID for collection isolation
      use_biobert: If True, use BioBERT for better medical document retrieval (default: False for Cloud Run)
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
            st.error("⚠️ API key not configured. Please contact administrator.")
            return None

        logger.info(f"Initializing QA system (use_biobert={use_biobert}, user_id={user_id})...")

        # Check if sentence-transformers is available
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(" sentence-transformers is available")
        except ImportError as e:
            logger.error(f" sentence-transformers not available: {e}")
            st.error("⚠️ Embedding library not available. Please contact administrator.")
            return None

        # Try to initialize with BioBERT first if requested, otherwise use default
        qa_system = None
        if use_biobert:
            try:
                logger.info("Attempting to initialize with BioBERT...")
                qa_system = FileQA(
                    gemini_api_key=api_key,
                    use_biobert=True,
                    use_vector_db=True,
                    user_id=user_id,
                    simplify_medical_terms=True,
                    privacy_mode=privacy_mode,
                    allow_external_calls=allow_external_calls,
                    pii_extra_terms=pii_extra_terms,
                )
                # Verify embedding model is loaded
                if qa_system.rag.embedding_model is None:
                    raise ValueError("BioBERT embedding model failed to load")
                logger.info(" QA system initialized with BioBERT successfully")
            except Exception as biobert_error:
                logger.warning(f"BioBERT initialization failed: {biobert_error}. Falling back to default model...")
                use_biobert = False

        if not qa_system:
            # Use default embedding model (all-MiniLM-L6-v2)
            logger.info("Initializing with default embedding model (all-MiniLM-L6-v2)...")
            qa_system = FileQA(
                gemini_api_key=api_key,
                use_biobert=False,  # Use default model
                use_vector_db=True,
                user_id=user_id,
                simplify_medical_terms=True,
                privacy_mode=privacy_mode,
                allow_external_calls=allow_external_calls,
                pii_extra_terms=pii_extra_terms,
            )
            # Verify embedding model is loaded
            if qa_system.rag.embedding_model is None:
                error_msg = "Default embedding model failed to load. Check logs for details."
                logger.error(error_msg)
                st.error(f"⚠️ {error_msg}")
                return None
            logger.info(" QA system initialized with default model successfully")

        return qa_system
    except Exception as e:
        logger.error(f"Failed to initialize QA system: {e}", exc_info=True)
        st.error(f"⚠️ System initialization failed: {str(e)}")
        return None


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def create_new_chat(uid: Optional[str], store: Optional[FirestoreStore]):
    """
    Create a new chat.

    - Signed-in: persist chat metadata to Firestore.
    - Not signed-in: store chat state in session only (cleared when session ends).
    """
    chat_id = str(uuid4())
    if uid and store:
        store.create_chat(uid, chat_id, title="New chat")
    else:
        st.session_state.local_chats.insert(
            0, {"chat_id": chat_id, "title": "New chat", "updated_at": datetime.utcnow().isoformat()}
        )
        st.session_state.local_messages_by_chat.setdefault(chat_id, [])
        st.session_state.local_docs_by_chat.pop(chat_id, None)
        st.session_state.local_files_by_chat[chat_id] = []
        st.session_state.local_docs_loaded_by_chat[chat_id] = False
    st.session_state.current_chat_id = chat_id
    st.session_state.qa_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.documents_loaded = False
    st.session_state.loaded_files = []
    st.session_state.show_file_upload = False
    st.session_state.load_success_message = None
    st.session_state.qa_system = initialize_qa_system(
        user_id=uid,
        privacy_mode=st.session_state.get("privacy_mode", True),
        allow_external_calls=st.session_state.get("allow_external_calls", True),
        pii_extra_terms=st.session_state.get("pii_extra_terms", []),
    )


def main():
    # --- Auth + global session keys ---
    if "user" not in st.session_state:
        st.session_state.user = None
    if "auth_error" not in st.session_state:
        st.session_state.auth_error = None
    if "auth_session_token" not in st.session_state:
        st.session_state.auth_session_token = ""
    if "persist_session_token" not in st.session_state:
        st.session_state.persist_session_token = ""
    if "clear_session_token" not in st.session_state:
        st.session_state.clear_session_token = False
    if "sid" not in st.session_state:
        st.session_state.sid = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "qa_chat_id" not in st.session_state:
        st.session_state.qa_chat_id = None
    if "show_file_upload" not in st.session_state:
        st.session_state.show_file_upload = False
    if "load_success_message" not in st.session_state:
        st.session_state.load_success_message = None
    if "privacy_mode" not in st.session_state:
        st.session_state.privacy_mode = True
    if "allow_external_calls" not in st.session_state:
        st.session_state.allow_external_calls = True
    if "pii_extra_terms" not in st.session_state:
        st.session_state.pii_extra_terms = []
    # Anonymous session storage (in-memory only)
    if "local_chats" not in st.session_state:
        st.session_state.local_chats = []
    if "local_messages_by_chat" not in st.session_state:
        st.session_state.local_messages_by_chat = {}
    if "local_docs_by_chat" not in st.session_state:
        st.session_state.local_docs_by_chat = {}
    if "local_files_by_chat" not in st.session_state:
        st.session_state.local_files_by_chat = {}
    if "local_docs_loaded_by_chat" not in st.session_state:
        st.session_state.local_docs_loaded_by_chat = {}

    # Keep a small bridge running so localStorage -> session_state works after refresh.
    render_auth_session_bridge()

    # If we need to persist/clear the session token in the browser, do it here.
    if st.session_state.get("clear_session_token"):
        components.html(
            """
<script>
  try {
    if (window.top && window.top.localStorage) {
      window.top.localStorage.removeItem("lab_lens_auth_session");
    } else {
      localStorage.removeItem("lab_lens_auth_session");
    }
  } catch (e) {}
</script>
""",
            height=0,
        )
        st.session_state.clear_session_token = False
        st.session_state.auth_session_token = ""

    if st.session_state.get("persist_session_token"):
        token_to_persist = st.session_state.persist_session_token
        components.html(
            f"""
<script>
  try {{
    if (window.top && window.top.localStorage) {{
      window.top.localStorage.setItem("lab_lens_auth_session", {json.dumps(token_to_persist)});
    }} else {{
      localStorage.setItem("lab_lens_auth_session", {json.dumps(token_to_persist)});
    }}
  }} catch (e) {{}}
</script>
""",
            height=0,
        )
        st.session_state.persist_session_token = ""

    # --- OAuth callback handling ---
    # If Google redirects back with ?code=...&state=..., complete the sign-in server-side.
    oauth_code = _get_query_param("code")
    oauth_state = _get_query_param("state")
    if oauth_code and oauth_state and not st.session_state.get("user"):
        cfg = load_google_oauth_config()
        if not cfg:
            st.session_state["auth_error"] = (
                "Google Sign-in is not configured. Set GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REDIRECT_URI."
            )
        else:
            payload = verify_state(oauth_state, cfg.client_secret)
            if not payload:
                st.session_state["auth_error"] = "Sign-in failed (invalid state). Please try again."
            else:
                try:
                    tokens = exchange_code_for_tokens(cfg, oauth_code)
                    idt = (tokens.get("id_token") or "").strip()
                    at = (tokens.get("access_token") or "").strip()
                    claims = verify_google_id_token(idt, client_id=cfg.client_id)

                    # Prefer userinfo for profile fields; fall back to id_token claims.
                    profile: Dict[str, Any] = {}
                    if at:
                        try:
                            profile = fetch_userinfo(at)
                        except Exception as e:
                            logger.warning(f"Failed to fetch Google userinfo: {e}")

                    uid = f"google:{claims.get('sub')}"
                    email = profile.get("email") or claims.get("email")
                    name = profile.get("name") or claims.get("name")
                    picture = profile.get("picture") or claims.get("picture")

                    fb_user = FirebaseUser(
                        uid=str(uid),
                        email=str(email) if email else None,
                        name=str(name) if name else None,
                        picture=str(picture) if picture else None,
                    )
                    st.session_state.user = fb_user
                    st.session_state.auth_error = None
                    # Persist a signed session token for refresh survival.
                    session_token = build_session_token(
                        {
                            "uid": fb_user.uid,
                            "email": fb_user.email,
                            "name": fb_user.name,
                            "picture": fb_user.picture,
                        },
                        cfg.client_secret,
                    )
                    st.session_state.auth_session_token = session_token
                    st.session_state.persist_session_token = session_token

                    # Additionally persist a stable session id in the URL + Firestore so refresh works
                    # even if browser storage is blocked by Streamlit iframe sandboxing.
                    try:
                        sid = _get_query_param("sid") or st.session_state.get("sid") or str(uuid4())
                        exp = int(time.time()) + (30 * 24 * 60 * 60)
                        get_firestore_store().upsert_session(
                            sid,
                            {
                                "uid": fb_user.uid,
                                "email": fb_user.email,
                                "name": fb_user.name,
                                "picture": fb_user.picture,
                            },
                            exp,
                        )
                        st.session_state.sid = sid
                    except Exception as e:
                        logger.warning(f"Failed to persist session id: {e}")

                    # Upsert user profile
                    try:
                        get_firestore_store().upsert_user(fb_user.uid, fb_user.email, fb_user.name, fb_user.picture)
                    except Exception as e:
                        logger.warning(f"Failed to upsert user profile: {e}")

                except Exception as e:
                    logger.warning(f"OAuth sign-in failed: {e}", exc_info=True)
                    st.session_state["auth_error"] = "Sign-in failed. Please try again."

        # Clear auth params from URL to avoid reprocessing, but keep `sid` if we set it.
        sid = _get_query_param("sid") or st.session_state.get("sid") or ""
        # Prefer explicit setter to ensure browser URL updates on Cloud Run.
        if sid:
            try:
                st.query_params.clear()  # type: ignore[attr-defined]
                st.query_params["sid"] = sid  # type: ignore[index]
            except Exception:
                pass
            st.experimental_set_query_params(sid=sid)
        else:
            try:
                st.query_params.clear()  # type: ignore[attr-defined]
            except Exception:
                pass
            st.experimental_set_query_params()
        st.rerun()

    # Restore user from Firestore-backed sid if present (survives refresh).
    if not st.session_state.get("user"):
        sid = _get_query_param("sid") or ""
        if sid:
            try:
                sess = get_firestore_store().get_session(sid)
                if sess and int(sess.get("exp", 0)) > int(time.time()):
                    u = (sess.get("user") or {}) if isinstance(sess.get("user"), dict) else {}
                    restored = FirebaseUser(
                        uid=str(u.get("uid") or ""),
                        email=str(u.get("email")) if u.get("email") else None,
                        name=str(u.get("name")) if u.get("name") else None,
                        picture=str(u.get("picture")) if u.get("picture") else None,
                    )
                    if restored.uid:
                        st.session_state.user = restored
                        st.session_state.sid = sid
            except Exception as e:
                logger.warning(f"Failed to restore session from Firestore: {e}")

    fb_user = ensure_user()
    if not fb_user:
        fb_user = st.session_state.get("user")

    # Ensure a stable sid is present in the URL whenever the user is signed in.
    # This makes refresh persistence robust even when browser storage is blocked.
    current_sid = _get_query_param("sid") or ""
    if fb_user and not current_sid:
        try:
            sid = st.session_state.get("sid") or str(uuid4())
            exp = int(time.time()) + (30 * 24 * 60 * 60)
            get_firestore_store().upsert_session(
                sid,
                {"uid": fb_user.uid, "email": fb_user.email, "name": fb_user.name, "picture": fb_user.picture},
                exp,
            )
            st.session_state.sid = sid
            try:
                st.query_params.clear()  # type: ignore[attr-defined]
                st.query_params["sid"] = sid  # type: ignore[index]
            except Exception:
                pass
            st.experimental_set_query_params(sid=sid)
            st.rerun()
        except Exception as e:
            logger.warning(f"Failed to set sid in URL: {e}")

    store = get_firestore_store() if fb_user else None

    # --- Sidebar: auth gate ---
    with st.sidebar:
        st.markdown("# 🏥 Lab Lens")
        st.markdown("---")

    # --- Load chats + current chat selection ---
    uid: Optional[str] = fb_user.uid if fb_user else None
    if uid and store:
        try:
            chats = store.list_chats(uid)
            chat_ids = {c["chat_id"] for c in chats}
            st.session_state.pop("firestore_error", None)
        except Exception as e:
            # Most common cause on a fresh GCP project: Firestore API not enabled / DB not created.
            # Fall back to local (non-persistent) chats so the app stays usable.
            logger.warning(f"Firestore unavailable; falling back to local session: {e}")
            st.session_state["firestore_error"] = str(e)
            store = None
            chats = st.session_state.local_chats
            chat_ids = {c.get("chat_id") for c in chats if c.get("chat_id")}
    else:
        chats = st.session_state.local_chats
        chat_ids = {c.get("chat_id") for c in chats if c.get("chat_id")}

    if not st.session_state.current_chat_id or st.session_state.current_chat_id not in chat_ids:
        if chats:
            st.session_state.current_chat_id = chats[0]["chat_id"]
        else:
            create_new_chat(uid, store)

    current_chat_id = st.session_state.current_chat_id

    # Initialize / refresh QA system when switching chats
    if st.session_state.get("qa_system") is None or st.session_state.qa_chat_id != current_chat_id:
        st.session_state.qa_system = initialize_qa_system(
            user_id=uid,
            privacy_mode=st.session_state.get("privacy_mode", True),
            allow_external_calls=st.session_state.get("allow_external_calls", True),
            pii_extra_terms=st.session_state.get("pii_extra_terms", []),
        )
        st.session_state.qa_chat_id = current_chat_id

        if uid and store:
            # Load persisted messages
            msgs = store.list_messages(uid, current_chat_id)
            st.session_state.messages = [
                {"role": m.get("role", "assistant"), "content": m.get("content", ""), "sources": m.get("sources", [])}
                for m in msgs
            ]

            # Load persisted chunks/embeddings and rebuild RAG index (if any)
            try:
                chunks, embeddings, metas = store.load_chunks(uid, current_chat_id)
                if chunks and embeddings and len(embeddings[0]) > 0:
                    st.session_state.qa_system.rag.load_cached_index(chunks, embeddings, metas)
                    st.session_state.documents_loaded = True
                    # Infer loaded file names (best-effort)
                    file_names = []
                    for meta in metas:
                        name = meta.get("document_name") or meta.get("file_name")
                        if name and name not in file_names:
                            file_names.append(name)
                    st.session_state.loaded_files = file_names
                else:
                    st.session_state.documents_loaded = False
                    st.session_state.loaded_files = []
            except Exception as e:
                logger.warning(f"Failed to load persisted document context: {e}")
                st.session_state.documents_loaded = False
                st.session_state.loaded_files = []
        else:
            # Anonymous session: restore in-memory chat/docs for this chat_id
            st.session_state.messages = st.session_state.local_messages_by_chat.get(current_chat_id, [])
            st.session_state.documents_loaded = bool(st.session_state.local_docs_loaded_by_chat.get(current_chat_id, False))
            st.session_state.loaded_files = st.session_state.local_files_by_chat.get(current_chat_id, [])
            payload = st.session_state.local_docs_by_chat.get(current_chat_id)
            try:
                if payload and payload.get("chunks") and payload.get("embeddings"):
                    st.session_state.qa_system.rag.load_cached_index(
                        payload["chunks"],
                        payload["embeddings"],
                        payload.get("metadata", []),
                    )
            except Exception as e:
                logger.warning(f"Failed to restore local document context: {e}")

    # Sidebar - persisted chats
    with st.sidebar:
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar", type="primary"):
            create_new_chat(uid, store)
            st.rerun()

        st.markdown("---")
        st.markdown("### 💬 Recent Chats")

        for chat in chats:
            chat_id = chat.get("chat_id")
            title = (chat.get("title") or chat_id or "Chat").strip()
            is_active = chat_id == st.session_state.current_chat_id
            display_name = title[:35] + "..." if len(title) > 35 else title
            button_style = "primary" if is_active else "secondary"
            if st.button(display_name, key=f"chat_{chat_id}", use_container_width=True, type=button_style):
                st.session_state.current_chat_id = chat_id
                st.session_state.show_file_upload = False
                # Force re-init to load messages/docs
                st.session_state.qa_chat_id = None
                st.rerun()

        st.markdown("---")
        if fb_user:
            st.markdown("### 👤 User")
            st.markdown(f"**{fb_user.name or fb_user.email or fb_user.uid}**")
            if fb_user.email:
                st.caption(fb_user.email)
            if st.session_state.get("firestore_error"):
                st.warning(
                    "Signed in, but chat persistence is temporarily unavailable because Firestore isn't enabled or "
                    "the service lacks permission. Enable Cloud Firestore in your GCP project and refresh."
                )
            if st.button("Sign out", use_container_width=True):
                # Clear URL session id + Firestore session mapping (best-effort).
                try:
                    sid = _get_query_param("sid") or st.session_state.get("sid") or ""
                    if sid:
                        get_firestore_store().delete_session(sid)
                except Exception as e:
                    logger.warning(f"Failed to delete session: {e}")
                try:
                    st.query_params.pop("sid", None)  # type: ignore[attr-defined]
                except Exception:
                    st.experimental_set_query_params()
                st.session_state.user = None
                st.session_state.auth_error = None
                st.session_state.clear_session_token = True
                st.session_state.sid = ""
                st.rerun()

        # Sign-in options at bottom (no persistence unless user signs in)
        if not fb_user:
            st.markdown("---")
            st.markdown("### 🔐 Sign in")
            render_google_sign_in()
            if st.session_state.get("auth_error"):
                st.error(f"Authentication error: {st.session_state['auth_error']}")

    # Show persistent status indicator if documents are loaded (at top)
    if st.session_state.documents_loaded:
        st.success(
            f"📄 **Documents loaded:** {', '.join(st.session_state.loaded_files[:3])}{' ...' if len(st.session_state.loaded_files) > 3 else ''} • Ready to answer questions!"
        )

    # Show success message if available
    if st.session_state.load_success_message:
        st.info(st.session_state.load_success_message)
        # Clear message after showing (so it doesn't persist forever)
        st.session_state.load_success_message = None

    # Main chat area - Clean ChatGPT-style
    if not st.session_state.messages:
        # Welcome screen (only show if no documents loaded or no messages)
        if not st.session_state.documents_loaded:
            st.markdown(
                """
      <div style="text-align: center; padding: 1rem 1rem 0.5rem;">
        <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.25rem;">🏥 Lab Lens</div>
        <div style="font-size: 1rem; color: #10a37f; margin-bottom: 0.75rem;">Your AI-Powered Medical Report Assistant</div>
        <p style="color: #b4b4b4; font-size: 0.9rem; margin-bottom: 1rem;">
          I help you understand complex medical lab reports by translating technical jargon into plain language.
        </p>
      </div>
      <div style="display: flex; gap: 1rem; max-width: 700px; margin: 0 auto; padding: 0 1rem;">
        <div style="flex: 1; background: rgba(16, 163, 127, 0.1); border-radius: 10px; padding: 1rem;">
          <p style="color: #ececec; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">✨ What I can do:</p>
          <ul style="color: #b4b4b4; margin: 0; padding-left: 1rem; font-size: 0.85rem; line-height: 1.6;">
            <li><strong>Explain</strong> lab results</li>
            <li><strong>Simplify</strong> medical terms</li>
            <li><strong>Summarize</strong> reports</li>
            <li><strong>Answer</strong> health questions</li>
          </ul>
        </div>
        <div style="flex: 1; background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 1rem; border: 1px solid #343541;">
          <p style="color: #ececec; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">🚀 Get Started:</p>
          <ol style="color: #b4b4b4; margin: 0; padding-left: 1rem; font-size: 0.85rem; line-height: 1.6;">
            <li>Click <strong style="color: #10a37f;">➕</strong> below</li>
            <li>Upload your lab report</li>
            <li>Click <strong>Load documents</strong></li>
            <li>Ask questions!</li>
          </ol>
        </div>
      </div>
      """,
                unsafe_allow_html=True,
            )
        else:
            # Show a welcome message for loaded documents
            st.markdown(
                """
      <div style="text-align: center; padding: 1.5rem;">
        <div style="font-size: 1.5rem; font-weight: bold;">📄 Documents Ready!</div>
        <div style="font-size: 1rem; color: #10a37f; margin: 0.5rem 0;">Your files have been processed</div>
        <p style="color: #888; font-size: 0.9rem;">Try: <em>"Summarize this report"</em> or <em>"What are the key findings?"</em></p>
      </div>
      """,
                unsafe_allow_html=True,
            )

    # Display chat messages in chronological order (oldest first, newest at bottom)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"][:3], 1):
                        score = source.get("score", 0)
                        raw_chunk = source.get("chunk", "")

                        # Clean PDF artifacts (cid:X codes)
                        import re

                        cleaned_chunk = re.sub(r"\(cid:\d+\)", " ", raw_chunk)
                        cleaned_chunk = re.sub(r"\s+", " ", cleaned_chunk).strip()

                        # Get a meaningful preview
                        preview = cleaned_chunk[:300] if cleaned_chunk else "[No text available]"

                        # Extract key terms from the user's question for highlighting
                        user_question = message.get("_question", "")

                        # Highlight relevant terms in the source
                        highlighted_preview = preview
                        if user_question:
                            # Extract important words (skip common words)
                            stop_words = {
                                "the",
                                "a",
                                "an",
                                "is",
                                "was",
                                "were",
                                "what",
                                "which",
                                "who",
                                "how",
                                "when",
                                "where",
                                "this",
                                "that",
                                "for",
                                "and",
                                "or",
                                "in",
                                "on",
                                "at",
                                "to",
                                "of",
                                "my",
                                "me",
                                "i",
                            }
                            key_words = [w for w in re.findall(r"\b\w{3,}\b", user_question.lower()) if w not in stop_words]

                            # Highlight matching words
                            for word in key_words[:5]:  # Limit to 5 key words
                                pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                                highlighted_preview = pattern.sub(r"**\1**", highlighted_preview)

                        st.caption(f"Source {i} (relevance: {score:.3f})")
                        st.markdown(f"> {highlighted_preview}{'...' if len(cleaned_chunk) > 300 else ''}")

    # File upload modal (only appears when + button is clicked)
    if st.session_state.show_file_upload:
        st.markdown("---")
        st.markdown("### 📁 Upload Files")
        col1, col2 = st.columns([2, 1])
        with col1:
            quick_upload = st.file_uploader(
                "Upload files",
                type=["txt", "pdf", "jpg", "jpeg", "png", "bmp", "tiff", "md"],
                accept_multiple_files=True,
                help="Upload text files, PDFs, or images",
                label_visibility="collapsed",
                key="quick_upload_main",
            )
        with col2:
            quick_text = st.text_area(
                "Or paste text",
                height=100,
                help="Paste raw text content",
                label_visibility="collapsed",
                placeholder="Paste text here...",
                key="quick_text_main",
            )

        col_load, col_close = st.columns([1, 1])
        with col_load:
            if st.button("📥 Load Documents", type="primary", use_container_width=True, key="load_main_btn"):
                if quick_upload or (quick_text and quick_text.strip()):
                    with st.spinner("Processing documents..."):
                        try:
                            if quick_upload:
                                file_paths = []
                                uploaded_uris = []
                                for uploaded_file in quick_upload:
                                    file_path = save_uploaded_file(uploaded_file)
                                    file_paths.append(file_path)
                                    # Persist original upload to GCS only if privacy mode is OFF (best-effort)
                                    if uid and store and (not st.session_state.get("privacy_mode", True)):
                                        gcs = get_gcs_store()
                                        if gcs:
                                            try:
                                                obj = gcs.upload_bytes(
                                                    uid=uid,
                                                    chat_id=current_chat_id,
                                                    filename=uploaded_file.name,
                                                    data=uploaded_file.getvalue(),
                                                    content_type=getattr(uploaded_file, "type", None),
                                                )
                                                uploaded_uris.append(obj.gs_uri)
                                            except Exception as e:
                                                logger.warning(f"GCS upload failed for {uploaded_file.name}: {e}")

                                result = st.session_state.qa_system.load_multiple_files(file_paths)
                                if result.get("success"):
                                    st.session_state.documents_loaded = True
                                    st.session_state.loaded_files = [f.name for f in quick_upload]
                                    st.session_state.show_file_upload = False
                                    # Persist chunks/embeddings for this chat (signed-in) OR keep in-memory (anonymous)
                                    try:
                                        payload = st.session_state.qa_system.rag.export_cached_index()
                                        if payload.get("chunks") and payload.get("embeddings"):
                                            chunks = payload["chunks"]
                                            metas = payload.get("metadata", [])
                                            if st.session_state.get("privacy_mode", True):
                                                chunks = [
                                                    redact_text(
                                                        c, extra_terms=st.session_state.get("pii_extra_terms", [])
                                                    ).text
                                                    for c in chunks
                                                ]
                                                safe_metas = []
                                                for m in metas:
                                                    mm = dict(m or {})
                                                    for key in ("document_name", "file_name"):
                                                        if key in mm and mm.get(key):
                                                            mm[key] = sanitize_filename(str(mm[key]))
                                                    safe_metas.append(mm)
                                                metas = safe_metas
                                            if uid and store:
                                                store.replace_chunks(
                                                    uid=uid,
                                                    chat_id=current_chat_id,
                                                    chunks=chunks,
                                                    embeddings=payload["embeddings"],
                                                    metadatas=metas,
                                                )
                                                store.update_chat(
                                                    uid,
                                                    current_chat_id,
                                                    doc_count=len(quick_upload),
                                                    files=[
                                                        (
                                                            sanitize_filename(f.name)
                                                            if st.session_state.get("privacy_mode", True)
                                                            else f.name
                                                        )
                                                        for f in quick_upload
                                                    ],
                                                    gcs_uris=uploaded_uris,
                                                )
                                            else:
                                                st.session_state.local_docs_by_chat[current_chat_id] = {
                                                    "chunks": chunks,
                                                    "embeddings": payload["embeddings"],
                                                    "metadata": metas,
                                                }
                                                st.session_state.local_docs_loaded_by_chat[current_chat_id] = True
                                                st.session_state.local_files_by_chat[current_chat_id] = [
                                                    f.name for f in quick_upload
                                                ]
                                    except Exception as e:
                                        logger.warning(f"Failed to persist document context: {e}")
                                    st.session_state.load_success_message = f" Successfully loaded {result['num_files']} file(s) ({result.get('num_chunks', 0)} chunks). Ready to answer questions!"
                                    st.rerun()
                                else:
                                    st.error(f" Failed to load files: {result.get('error', 'Unknown error')}")
                                    st.session_state.load_success_message = None

                            if quick_text and quick_text.strip():
                                result = st.session_state.qa_system.load_text(quick_text.strip())
                                if result.get("success"):
                                    st.session_state.documents_loaded = True
                                    if "Text Input" not in st.session_state.loaded_files:
                                        st.session_state.loaded_files.append("Text Input")
                                    st.session_state.show_file_upload = False
                                    # Persist chunks/embeddings for this chat (text input overwrites current context)
                                    try:
                                        payload = st.session_state.qa_system.rag.export_cached_index()
                                        if payload.get("chunks") and payload.get("embeddings"):
                                            chunks = payload["chunks"]
                                            metas = payload.get("metadata", [])
                                            if st.session_state.get("privacy_mode", True):
                                                chunks = [
                                                    redact_text(
                                                        c, extra_terms=st.session_state.get("pii_extra_terms", [])
                                                    ).text
                                                    for c in chunks
                                                ]
                                            if uid and store:
                                                store.replace_chunks(
                                                    uid=uid,
                                                    chat_id=current_chat_id,
                                                    chunks=chunks,
                                                    embeddings=payload["embeddings"],
                                                    metadatas=metas,
                                                )
                                                store.update_chat(uid, current_chat_id, doc_count=1, files=["Text Input"])
                                            else:
                                                st.session_state.local_docs_by_chat[current_chat_id] = {
                                                    "chunks": chunks,
                                                    "embeddings": payload["embeddings"],
                                                    "metadata": metas,
                                                }
                                                st.session_state.local_docs_loaded_by_chat[current_chat_id] = True
                                                st.session_state.local_files_by_chat[current_chat_id] = ["Text Input"]
                                    except Exception as e:
                                        logger.warning(f"Failed to persist text context: {e}")
                                    st.session_state.load_success_message = f" Successfully loaded text ({result.get('num_chunks', 0)} chunks). Ready to answer questions!"
                                    st.rerun()
                                else:
                                    st.error(f" Failed to load text: {result.get('error', 'Unknown error')}")
                                    st.session_state.load_success_message = None
                        except Exception as e:
                            st.error(f" Error: {e}")
                else:
                    st.warning("⚠️ Please upload a file or paste text")

        with col_close:
            if st.button(" Close", use_container_width=True, key="close_main_btn"):
                st.session_state.show_file_upload = False
                st.rerun()

        st.markdown("---")

    # Add spacing at bottom for fixed input
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Fixed chat input at bottom - always visible
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)

    # Use columns: + button, input
    col_add, col_input = st.columns([0.05, 0.95], gap="small")

    with col_add:
        # + button styled to connect with input
        if st.button("➕", help="Add files", key="add_files_button", use_container_width=True):
            st.session_state.show_file_upload = not st.session_state.show_file_upload
            st.rerun()

    with col_input:
        chat_placeholder = (
            "Ask a question about your documents..."
            if st.session_state.documents_loaded
            else "Upload files first to ask questions..."
        )

        if prompt := st.chat_input(chat_placeholder, key="chat_input_main"):
            # Check if documents are loaded
            if not st.session_state.documents_loaded:
                st.warning(
                    "⚠️ **Documents not loaded yet!** Please:\n1. Click ➕ button to upload files\n2. Click '📥 Load Documents' button to process them\n3. Then ask your question"
                )
                st.stop()

            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            to_store = (
                redact_text(prompt, extra_terms=st.session_state.get("pii_extra_terms", [])).text
                if st.session_state.get("privacy_mode", True)
                else prompt
            )
            if uid and store:
                try:
                    store.add_message(uid, current_chat_id, role="user", content=to_store)
                    # If this is the first user message, use it as chat title
                    if len([m for m in st.session_state.messages if m.get("role") == "user"]) == 1:
                        title_seed = to_store
                        title = title_seed[:60] + ("..." if len(title_seed) > 60 else "")
                        store.update_chat(uid, current_chat_id, title=title)
                except Exception as e:
                    logger.warning(f"Failed to persist user message: {e}")
            else:
                # Anonymous/local: keep messages and title in session state only.
                st.session_state.local_messages_by_chat[current_chat_id] = list(st.session_state.messages)
                # Set title on first user message
                if len([m for m in st.session_state.messages if m.get("role") == "user"]) == 1:
                    title = to_store[:60] + ("..." if len(to_store) > 60 else "")
                    for c in st.session_state.local_chats:
                        if c.get("chat_id") == current_chat_id:
                            c["title"] = title
                            c["updated_at"] = datetime.utcnow().isoformat()
                            break

            # Check if user wants a summary (use MedicalSummarizer)
            prompt_lower = prompt.lower().strip()
            is_summary_request = any(
                keyword in prompt_lower
                for keyword in [
                    "summarize",
                    "summary",
                    "summarise",
                    "summarise this",
                    "give me a summary",
                    "create a summary",
                    "generate summary",
                ]
            )

            # Generate response
            with st.spinner("Generating summary using MedicalSummarizer..." if is_summary_request else "Thinking..."):
                try:
                    if is_summary_request:
                        # Use MedicalSummarizer for summarization
                        result = st.session_state.qa_system.summarize_document()

                        if result.get("success"):
                            answer = result.get("summary", "Summary not available")
                            sources = []  # Summarizer doesn't return sources in the same format
                        else:
                            # Fallback to Gemini if summarizer fails
                            error_msg = result.get("error", "Unknown error")
                            logger.warning(f"Summarizer failed: {error_msg}. Falling back to Gemini.")
                            result = st.session_state.qa_system.ask_question(prompt)
                            answer = result.get("answer", "No answer available")
                            sources = result.get("sources", [])
                    else:
                        # Use Gemini for Q&A
                        result = st.session_state.qa_system.ask_question(prompt)

                        answer = result.get("answer", "No answer available")
                        sources = result.get("sources", [])

                    # Add assistant response to chat (include question for source highlighting)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources, "_question": prompt}
                    )
                    to_store_answer = (
                        redact_text(answer, extra_terms=st.session_state.get("pii_extra_terms", [])).text
                        if st.session_state.get("privacy_mode", True)
                        else answer
                    )
                    to_store_sources = (
                        redact_sources(sources, extra_terms=st.session_state.get("pii_extra_terms", []))
                        if st.session_state.get("privacy_mode", True)
                        else sources
                    )
                    if uid and store:
                        try:
                            store.add_message(
                                uid,
                                current_chat_id,
                                role="assistant",
                                content=to_store_answer,
                                sources=to_store_sources,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to persist assistant message: {e}")
                    else:
                        st.session_state.local_messages_by_chat[current_chat_id] = list(st.session_state.messages)
                        for c in st.session_state.local_chats:
                            if c.get("chat_id") == current_chat_id:
                                c["updated_at"] = datetime.utcnow().isoformat()
                                break

                    # Rerun to display new messages and scroll to bottom
                    st.rerun()

                except Exception as e:
                    error_msg = f" Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Scroll to bottom button
    st.markdown(
        """
  <button class="scroll-to-bottom-btn" id="scrollToBottomBtn" onclick="scrollToBottom()" title="Scroll to bottom">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M19 12l-7 7-7-7M19 5l-7 7-7-7"/>
    </svg>
  </button>
  """,
        unsafe_allow_html=True,
    )

    # JavaScript for auto-scroll and scroll detection
    st.markdown(
        """
  <script type="text/javascript">
    let autoScrollEnabled = true;
    let userScrolledUp = false;
    let lastMessageCount = 0;
   
    function scrollToBottom() {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
      });
      autoScrollEnabled = true;
      userScrolledUp = false;
      updateScrollButton();
    }
   
    function updateScrollButton() {
      const btn = document.getElementById('scrollToBottomBtn');
      if (!btn) return;
     
      const scrollHeight = document.documentElement.scrollHeight;
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const clientHeight = document.documentElement.clientHeight;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
     
      // Show button if user is more than 200px from bottom
      if (distanceFromBottom > 200) {
        btn.classList.add('visible');
      } else {
        btn.classList.remove('visible');
      }
    }
   
    function checkForNewMessages() {
      const currentMessageCount = document.querySelectorAll('[data-testid="stChatMessage"]').length;
     
      // If new messages were added, auto-scroll if user hasn't scrolled up
      if (currentMessageCount > lastMessageCount && !userScrolledUp) {
        setTimeout(() => {
          scrollToBottom();
        }, 100);
      }
     
      lastMessageCount = currentMessageCount;
    }
   
    // Auto-scroll on page load
    window.addEventListener('load', function() {
      setTimeout(() => {
        scrollToBottom();
        checkForNewMessages();
      }, 300);
    });
   
    // Detect user scrolling
    let scrollTimeout;
    window.addEventListener('scroll', function() {
      clearTimeout(scrollTimeout);
     
      const scrollHeight = document.documentElement.scrollHeight;
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const clientHeight = document.documentElement.clientHeight;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
     
      // If user is more than 100px from bottom, they've scrolled up
      if (distanceFromBottom > 100) {
        userScrolledUp = true;
        autoScrollEnabled = false;
      } else {
        // User is near bottom, re-enable auto-scroll
        userScrolledUp = false;
        autoScrollEnabled = true;
      }
     
      updateScrollButton();
     
      // Check for new messages after scroll settles
      scrollTimeout = setTimeout(checkForNewMessages, 100);
    });
   
    // Monitor for new messages (MutationObserver)
    const observer = new MutationObserver(function(mutations) {
      checkForNewMessages();
    });
   
    // Start observing when DOM is ready
    if (document.body) {
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    } else {
      document.addEventListener('DOMContentLoaded', function() {
        observer.observe(document.body, {
          childList: true,
          subtree: true
        });
      });
    }
   
    // Periodic check for new messages (fallback)
    setInterval(checkForNewMessages, 500);
  </script>
  """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
