"""
rag.py — RAG (Retrieval-Augmented Generation) module for Meeting Scheduler.

Indexes the user's past Google Calendar events into a local FAISS vector store
using Google Generative AI Embeddings. The agent calls retrieve_context() before
answering questions or scheduling, injecting relevant historical context such as
recurring attendees, typical durations, and preferred time slots.
"""

import os
import warnings
# Suppress langchain-community sunset deprecation notice in CLI output
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
import datetime
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TZ = ZoneInfo("Asia/Kolkata")
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calendar_index")
DAYS_TO_INDEX = 90          # How far back to pull events
MAX_EVENTS_TO_INDEX = 500   # Safety cap


# ─────────────────────────────────────────────
# Build Documents from Calendar Events
# ─────────────────────────────────────────────
def _events_to_documents(items: list) -> list[Document]:
    """
    Convert raw Google Calendar event dicts into LangChain Documents.
    Each document captures the meeting title, date, time, duration,
    attendees, and description — all information useful for context retrieval.
    """
    docs = []
    for ev in items:
        try:
            start_raw = ev["start"].get("dateTime", ev["start"].get("date", ""))
            end_raw   = ev["end"].get("dateTime",   ev["end"].get("date", ""))

            # Parse times
            if "T" in start_raw:
                start_raw = start_raw.replace("Z", "+00:00")
                end_raw   = end_raw.replace("Z",   "+00:00")
                ev_start  = datetime.datetime.fromisoformat(start_raw).astimezone(TZ)
                ev_end    = datetime.datetime.fromisoformat(end_raw).astimezone(TZ)
            else:
                d         = datetime.date.fromisoformat(start_raw)
                ev_start  = datetime.datetime(d.year, d.month, d.day, 0, 0, tzinfo=TZ)
                ev_end    = ev_start + datetime.timedelta(hours=1)

            duration_min = int((ev_end - ev_start).total_seconds() / 60)

            # Attendees
            raw_attendees = ev.get("attendees", [])
            attendee_emails = [a.get("email", "") for a in raw_attendees if a.get("email")]

            title       = ev.get("summary", "Untitled Meeting")
            description = ev.get("description", "")
            location    = ev.get("location", "")

            # Rich text content for embedding
            content = (
                f"Meeting: {title}\n"
                f"Date: {ev_start.strftime('%A, %Y-%m-%d')}\n"
                f"Time: {ev_start.strftime('%I:%M %p')} – {ev_end.strftime('%I:%M %p')}\n"
                f"Duration: {duration_min} minutes\n"
                f"Attendees: {', '.join(attendee_emails) if attendee_emails else 'None listed'}\n"
                f"Location: {location if location else 'N/A'}\n"
                f"Description: {description[:300] if description else 'N/A'}"
            )

            docs.append(Document(
                page_content=content,
                metadata={
                    "title":      title,
                    "date":       ev_start.strftime("%Y-%m-%d"),
                    "weekday":    ev_start.strftime("%A"),
                    "start_time": ev_start.strftime("%H:%M"),
                    "duration":   duration_min,
                    "attendees":  ", ".join(attendee_emails),
                }
            ))
        except Exception:
            # Skip malformed events silently
            continue

    return docs


# ─────────────────────────────────────────────
# Build / Load Vector Store
# ─────────────────────────────────────────────
def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """
    Fetch past calendar events and index them into a FAISS vector store.
    Saves the index to disk at `calendar_index/` for reuse.

    Args:
        force_rebuild: If True, always rebuild even if index already exists.

    Returns:
        A loaded FAISS vector store instance.
    """
    # Lazy import to avoid circular dependency with tools.py
    from tools import get_calendar_service

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not force_rebuild and os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("[RAG] Loading existing calendar index...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("[RAG] Building calendar index from past events...")
    service  = get_calendar_service()
    now      = datetime.datetime.now(tz=TZ)
    past_90  = now - datetime.timedelta(days=DAYS_TO_INDEX)

    events_result = service.events().list(
        calendarId  = "primary",
        timeMin     = past_90.isoformat(),
        timeMax     = now.isoformat(),
        maxResults  = MAX_EVENTS_TO_INDEX,
        singleEvents= True,
        orderBy     = "startTime",
    ).execute()

    items = events_result.get("items", [])

    if not items:
        # If no history, create a tiny placeholder so the store is still valid
        docs = [Document(
            page_content="No past meeting history available yet.",
            metadata={"title": "placeholder"}
        )]
    else:
        docs = _events_to_documents(items)

    store = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    store.save_local(INDEX_DIR)
    print(f"[RAG] Indexed {len(docs)} past events → saved to {INDEX_DIR}/")
    return store


def load_or_build_store() -> FAISS:
    """Load the FAISS store if it exists, otherwise build it."""
    return build_vector_store(force_rebuild=False)


# ─────────────────────────────────────────────
# Context Retrieval
# ─────────────────────────────────────────────
_store: FAISS | None = None


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Retrieve the top-k most relevant past meetings for the given query.

    Args:
        query: The user's natural language request.
        k:     Number of past events to retrieve.

    Returns:
        A formatted string summarising relevant past meetings, ready to inject
        into the agent's system prompt as contextual background.
    """
    global _store
    if _store is None:
        _store = load_or_build_store()

    results = _store.similarity_search(query, k=k)

    if not results:
        return ""

    lines = ["📚 Relevant past meetings from your calendar history:"]
    for doc in results:
        lines.append(f"  • {doc.page_content.splitlines()[0]}")  # Meeting title line
        for line in doc.page_content.splitlines()[1:]:
            lines.append(f"    {line}")
        lines.append("")

    return "\n".join(lines)


def refresh_index() -> str:
    """
    Force-rebuild the RAG index (e.g., after a batch of new events are created).
    Returns a confirmation message.
    """
    global _store
    _store = build_vector_store(force_rebuild=True)
    return "[RAG] Calendar index refreshed successfully."
