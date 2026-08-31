import os
import datetime
from dotenv import load_dotenv

# Explicitly load .env from the same directory as this file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from tools import (
    create_event,
    get_calendar_events,
    find_free_slots,
    analyse_booking_patterns,
    query_calendar_insights,
    extract_meeting_from_image,
)
from rag import retrieve_context, load_or_build_store

# ─────────────────────────────────────────────
# All tools registered
# ─────────────────────────────────────────────
ALL_TOOLS = [
    create_event,
    get_calendar_events,
    find_free_slots,
    analyse_booking_patterns,
    query_calendar_insights,
    extract_meeting_from_image,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ─────────────────────────────────────────────
# Base System Prompt
# ─────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are an intelligent AI meeting scheduler with access to Google Calendar.

Today's date is {today}. Use this as the reference for all relative dates like
"tomorrow", "next Monday", "this Friday", etc.

Your capabilities:
1. CREATE meetings — parse natural language, extract details, and call create_event.
2. CHECK calendar — fetch events on any date with get_calendar_events.
3. DETECT conflicts — create_event automatically checks for overlaps and rejects past dates.
4. SUGGEST alternatives — when a slot is blocked:
   a. Call analyse_booking_patterns to understand the user's scheduling habits.
   b. Call find_free_slots on the originally requested day.
   c. Call find_free_slots on the user's lightest day(s).
   d. Present 2-3 ranked alternatives with a reason for each.
5. ANSWER calendar questions — use query_calendar_insights for open-ended questions
   like "How many hours of meetings this week?" or "Which days am I free?".
6. EXTRACT from images — if the user provides an image path (e.g., a WhatsApp or email
   screenshot), call extract_meeting_from_image(image_path) first to extract meeting
   details, then immediately call create_event with those details.

Always extract: title, date (YYYY-MM-DD), start_time (HH:MM 24h), duration_minutes.
If the user says "tomorrow", compute the actual date based on today ({today}).
If they say "3pm", convert to 15:00.
If the meeting title is not provided, use a sensible default like "Call" or "Meeting".
If duration is not provided, default to 30 minutes.
Do NOT ask the user for missing fields — infer or use defaults and proceed immediately.
If date is ambiguous, ask for clarification before proceeding.

When a conflict is found, always follow up with smart alternative suggestions.
Be concise, helpful, and proactive.

{rag_context}"""


# ─────────────────────────────────────────────
# Agent Runner
# ─────────────────────────────────────────────
def create_scheduler_agent():
    today = datetime.date.today().isoformat()

    # Warm up RAG index on startup (loads from disk if already built)
    print("[RAG] Initializing calendar knowledge base...")
    try:
        load_or_build_store()
        print("[RAG] Knowledge base ready.\n")
    except Exception as e:
        print(f"[RAG] Warning: Could not build knowledge base: {e}\n")

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def run_agent(user_input: str) -> str:
        # ── Retrieve RAG context for this specific query ──────────
        rag_context = ""
        try:
            raw = retrieve_context(user_input, k=4)
            if raw and "No past meeting" not in raw:
                rag_context = (
                    "\n--- HISTORICAL CONTEXT FROM YOUR PAST MEETINGS ---\n"
                    + raw +
                    "\n--- Use the above context to personalise your response ---\n"
                )
        except Exception:
            pass  # RAG failure should never block the agent

        # ── Build the system prompt with today + RAG context ──────
        system_prompt = BASE_SYSTEM_PROMPT.format(
            today=today,
            rag_context=rag_context,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        MAX_ITERATIONS = 10  # safety cap to prevent infinite loops

        for iteration in range(MAX_ITERATIONS):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, we have the final answer
            if not response.tool_calls:
                # Handle content being list or string
                content = response.content
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    ).strip()
                return content

            # Execute all tool calls and collect results
            print(f"\n[Agent] Iteration {iteration + 1}: executing {len(response.tool_calls)} tool call(s)...")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id   = tool_call["id"]

                print(f"  -> Tool: {tool_name}  Args: {tool_args}")

                if tool_name not in TOOL_MAP:
                    result = f"Error: Unknown tool '{tool_name}'"
                else:
                    try:
                        result = TOOL_MAP[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Error executing {tool_name}: {str(e)}"

                print(f"  <- Result: {str(result)[:120]}{'...' if len(str(result)) > 120 else ''}")

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

        return "I wasn't able to complete the request after multiple attempts. Please try rephrasing."

    return run_agent