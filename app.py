"""
app.py — Streamlit Web UI for AI Meeting Scheduler
Run with: streamlit run app.py
"""

import os
import tempfile
import datetime

import streamlit as st
from dotenv import load_dotenv

# ── Load env vars ──────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Meeting Scheduler",
    page_icon="🗓️",
    layout="wide",
)

# ── Custom CSS for a cleaner look ──────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .status-ok  { color: #22c55e; font-weight: 600; }
    .status-err { color: #ef4444; font-weight: 600; }
    .status-warn{ color: #f59e0b; font-weight: 600; }
    div[data-testid="stSidebarContent"] { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH  = os.path.join(PROJECT_DIR, "token.json")
INDEX_DIR   = os.path.join(PROJECT_DIR, "calendar_index")


def calendar_connected() -> bool:
    return os.path.exists(TOKEN_PATH)


def rag_indexed() -> bool:
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content}]

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_error" not in st.session_state:
    st.session_state.agent_error = None


@st.cache_resource(show_spinner="🤖 Initialising AI agent…")
def load_agent():
    """Load agent once and cache for the session."""
    from agent import create_scheduler_agent
    return create_scheduler_agent()


def get_agent():
    if st.session_state.agent is None:
        try:
            st.session_state.agent = load_agent()
            st.session_state.agent_error = None
        except Exception as e:
            st.session_state.agent_error = str(e)
    return st.session_state.agent


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🗓️ AI Meeting Scheduler")
    st.caption("Powered by Gemini + LangChain")
    st.divider()

    # ── Calendar connection status ─────────────────────────────────────────────
    st.subheader("📅 Google Calendar")
    if calendar_connected():
        st.markdown('<p class="status-ok">✅ Connected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">❌ Not connected</p>', unsafe_allow_html=True)
        st.caption("Run `python3 main.py` once to complete OAuth, then restart this app.")

    st.divider()

    # ── RAG index status ───────────────────────────────────────────────────────
    st.subheader("🧠 RAG Knowledge Base")
    if rag_indexed():
        st.markdown('<p class="status-ok">✅ Index ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-warn">⚠️ Not built yet</p>', unsafe_allow_html=True)
        st.caption("Will auto-build on first message.")

    if st.button("🔄 Refresh Index", use_container_width=True):
        with st.spinner("Rebuilding calendar index…"):
            try:
                from rag import refresh_index
                refresh_index()
                st.success("Index refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # ── Image upload ───────────────────────────────────────────────────────────
    st.subheader("📸 Schedule from Image")
    st.caption("Upload a WhatsApp/email screenshot to auto-extract meeting details.")
    uploaded_file = st.file_uploader(
        "Drop image here",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        label_visibility="collapsed",
    )

    extra_instruction = st.text_input(
        "Extra instruction (optional)",
        placeholder="e.g. Decline if I'm busy",
    )

    if st.button("📤 Extract & Schedule", use_container_width=True, disabled=uploaded_file is None):
        if uploaded_file:
            # Save to a temp file so the tool can read it by path
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            if extra_instruction.strip():
                prompt = (
                    f"The user has provided an image at path '{tmp_path}'. "
                    f"Call extract_meeting_from_image with this path to extract the meeting details, "
                    f"then act on this instruction: {extra_instruction}"
                )
            else:
                prompt = (
                    f"The user has provided an image at path '{tmp_path}'. "
                    f"Call extract_meeting_from_image with this path to extract the meeting details, "
                    f"then schedule the meeting by calling create_event with the extracted fields."
                )

            st.session_state.messages.append({"role": "user", "content": f"📸 Image uploaded: `{uploaded_file.name}`{' — ' + extra_instruction if extra_instruction else ''}"})
            st.session_state["_pending_prompt"] = prompt
            st.rerun()

    st.divider()

    # ── Utility buttons ────────────────────────────────────────────────────────
    st.subheader("⚙️ Options")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Today: **{datetime.date.today().strftime('%A, %d %b %Y')}**")


# ══════════════════════════════════════════════════════════════════════════════
# Main Chat Area
# ══════════════════════════════════════════════════════════════════════════════

st.header("💬 Chat with your Calendar", divider="gray")

# Show example prompts if chat is empty
if not st.session_state.messages:
    st.markdown("#### Try these prompts:")
    cols = st.columns(2)
    examples = [
        "📅 Schedule a 1-hour meeting called Team Sync tomorrow at 10am",
        "🔍 Am I free next Tuesday at 2pm?",
        "📊 How many hours of meetings do I have this week?",
        "🗓️ Which days am I free this week?",
        "⏰ Book a 30-min standup at 9am this Friday",
        "📈 What was my busiest day this month?",
    ]
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                # Strip emoji prefix for the actual prompt
                clean = ex.split(" ", 1)[1]
                st.session_state.messages.append({"role": "user", "content": ex})
                st.session_state["_pending_prompt"] = clean
                st.rerun()

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ── Handle pending prompt (from sidebar image upload or example button) ────────
pending = st.session_state.pop("_pending_prompt", None)

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your request… (e.g. Schedule a 1-hr meeting tomorrow at 3pm)")

prompt_to_run = pending or (user_input.strip() if user_input else None)

if prompt_to_run:
    # If it came from the text box (not a pending prompt), show user message
    if user_input and not pending:
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input.strip())

    # Run agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            agent = get_agent()

            if st.session_state.agent_error:
                reply = f"❌ **Agent failed to load:** {st.session_state.agent_error}\n\nMake sure your `.env` file has `GOOGLE_API_KEY` set."
            elif not calendar_connected():
                reply = (
                    "❌ **Google Calendar not connected.**\n\n"
                    "Run `python3 main.py` in your terminal once to complete the OAuth flow, "
                    "then come back here."
                )
            else:
                try:
                    reply = agent(prompt_to_run)
                except Exception as e:
                    reply = f"⚠️ Error: {e}"

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
