# AI Meeting Scheduler — LangChain + Google Calendar + Gemini

## Setup

### 1. Clone and create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install langchain langchain-google-genai langchain-community \
            faiss-cpu Pillow \
            google-auth google-auth-oauthlib google-api-python-client \
            python-dotenv
```

### 3. Get Gemini API Key
- Go to https://aistudio.google.com/app/apikey
- Create a key, then add to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

### 4. Google Calendar API Setup
1. Go to https://console.cloud.google.com
2. Create a new project
3. Enable **Google Calendar API** (APIs & Services → Library)
4. Create **OAuth 2.0 Client ID** (Desktop App) → download → rename to `credentials.json`
5. OAuth consent screen → Add your Gmail as a **Test User**

### 5. First Run (OAuth)
```bash
python main.py
```
- A URL is printed — open it in your browser
- Log in, grant Calendar permissions, copy the code
- Paste it back in the terminal
- `token.json` is saved — future runs skip this step
- A `calendar_index/` directory is created automatically (RAG index of your past events)

## Usage

```bash
python main.py
```

### Text prompts
- `Schedule a 1-hour meeting called Team Sync tomorrow at 10am`
- `Book a 30-min standup at 9am this Friday`
- `Set up a 45-min call with raj@example.com on Monday at 3pm`
- `How many hours of meetings do I have this week?`
- `Which days am I free this week?`
- `What was my busiest day this month?`
- `Who usually attends my standups?` ← answered using RAG history

### 📸 Multi-Modal — Schedule from a screenshot
```
image /path/to/whatsapp_invite.png
image /path/to/email_screenshot.jpg Schedule this for me
```
Gemini Vision reads the image, extracts the meeting title, date, time, and attendee email, then schedules it automatically.

### 🧠 RAG — Contextual scheduling
Past calendar events (last 90 days) are indexed into a local FAISS vector store on first run.
For every query, the 4 most relevant past meetings are retrieved and injected into the agent context, enabling smarter suggestions based on your habits.

To force-refresh the index after many new events:
```python
from rag import refresh_index
refresh_index()
```

## Architecture

| File | Purpose |
|------|---------|
| `tools.py` | All LangChain tools (create, check, find, analyse, insights, **image extraction**) |
| `agent.py` | Multi-step agent loop with tool execution + **RAG context injection** |
| `rag.py` | **NEW** — FAISS vector store builder and retrieval for past calendar events |
| `main.py` | CLI entry point with **image input** prefix detection |
| `calendar_index/` | **Auto-generated** — local FAISS index of your past meetings |

## Tasks Covered

| Task | Marks | What's implemented |
|------|-------|-------------------|
| Task 1 | 10 | OAuth setup, Calendar ID, token flow |
| Task 2 | 30 | NL → LLM → create_event → Calendar |
| Task 3 | 25 | Past-date guard, overlap check, agent loop |
| Task 4 | 35 | find_free_slots, analyse_booking_patterns, smart suggestions |
| Bonus | 20 | query_calendar_insights — free days, hours, busiest day |
| **GenAI+** | — | **Multi-Modal input** (Gemini Vision) + **RAG** (FAISS + Google Embeddings) |