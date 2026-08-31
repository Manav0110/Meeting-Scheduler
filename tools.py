import os
import base64
import datetime
from zoneinfo import ZoneInfo

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SCOPES = ['https://www.googleapis.com/auth/calendar']
TZ = ZoneInfo("Asia/Kolkata")
WORK_START = 9   # 9 AM
WORK_END   = 18  # 6 PM


# ─────────────────────────────────────────────
# Google Calendar Auth
# ─────────────────────────────────────────────
def get_calendar_service():
    """Authenticate and return Google Calendar API service."""
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            auth_url, _ = flow.authorization_url(prompt='consent')
            print("\n→ Open this URL in your browser:\n")
            print(auth_url)
            code = input("\n→ Paste the authorization code here: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds)


# ─────────────────────────────────────────────
# Helper: Parse event times  (Python 3.10 safe)
# ─────────────────────────────────────────────
def parse_event_time(time_str: str) -> datetime.datetime:
    """Parse ISO datetime string to aware datetime in IST.
    Handles both Z suffix and +HH:MM offsets (Python 3.10 compatible)."""
    if 'T' in time_str:
        # Python 3.10 cannot parse 'Z' — replace with +00:00
        time_str = time_str.replace('Z', '+00:00')
        dt = datetime.datetime.fromisoformat(time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
        return dt.astimezone(TZ)
    else:
        # All-day event
        d = datetime.date.fromisoformat(time_str)
        return datetime.datetime(d.year, d.month, d.day, 0, 0, tzinfo=TZ)


# ─────────────────────────────────────────────
# TOOL: get_calendar_events  (Task 3)
# ─────────────────────────────────────────────
@tool
def get_calendar_events(date: str) -> str:
    """
    Fetch all events from Google Calendar on a specific date.

    Args:
        date: Date in YYYY-MM-DD format

    Returns:
        A readable string listing all events with their start and end times,
        or a message saying the day is free.
    """
    service = get_calendar_service()

    day = datetime.date.fromisoformat(date)
    time_min = datetime.datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=TZ).isoformat()
    time_max = datetime.datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=TZ).isoformat()

    events_result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])

    if not events:
        return f"No events on {date}. The day is completely free."

    lines = [f"Events on {date}:"]
    for ev in events:
        summary = ev.get('summary', 'Untitled')
        start = parse_event_time(ev['start'].get('dateTime', ev['start'].get('date')))
        end   = parse_event_time(ev['end'].get('dateTime', ev['end'].get('date')))
        lines.append(f"  • {summary}: {start.strftime('%I:%M %p')} → {end.strftime('%I:%M %p')}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOL: create_event  (Task 2 + Task 3 guards)
# ─────────────────────────────────────────────
@tool
def create_event(
    title: str,
    date: str,
    start_time: str,
    duration_minutes: int,
    attendee_email: str = None
) -> str:
    """
    Creates a meeting in Google Calendar after validating the slot.

    Guards applied before creation:
      1. Rejects meetings scheduled in the past.
      2. Rejects if the slot overlaps with an existing event.

    Args:
        title: Meeting title
        date: YYYY-MM-DD
        start_time: HH:MM (24-hour format)
        duration_minutes: Duration of the meeting in minutes
        attendee_email: Optional attendee email address

    Returns:
        Event link on success, or an error message describing the problem.
    """
    # ── Parse requested times ──────────────────────────────────────
    try:
        new_start = datetime.datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        new_start = new_start.replace(tzinfo=TZ)
    except ValueError:
        return "Error: Invalid date or time format. Use YYYY-MM-DD for date and HH:MM for time."

    new_end = new_start + datetime.timedelta(minutes=duration_minutes)

    # ── Guard 1: Past date check ───────────────────────────────────
    now = datetime.datetime.now(tz=TZ)
    if new_start < now:
        return (
            f"Cannot schedule '{title}' at {new_start.strftime('%Y-%m-%d %H:%M')} — "
            f"that time is in the past. Please provide a future date and time."
        )

    # ── Guard 2: Overlap check ─────────────────────────────────────
    service = get_calendar_service()
    day = datetime.date.fromisoformat(date)
    time_min = datetime.datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=TZ).isoformat()
    time_max = datetime.datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=TZ).isoformat()

    events_result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    existing_events = events_result.get('items', [])

    for ev in existing_events:
        ev_start = parse_event_time(ev['start'].get('dateTime', ev['start'].get('date')))
        ev_end   = parse_event_time(ev['end'].get('dateTime', ev['end'].get('date')))
        ev_name  = ev.get('summary', 'Untitled')

        # Overlap condition: new_start < ev_end AND new_end > ev_start
        if new_start < ev_end and new_end > ev_start:
            return (
                f"Conflict detected: '{title}' ({new_start.strftime('%I:%M %p')}–{new_end.strftime('%I:%M %p')}) "
                f"overlaps with existing event '{ev_name}' "
                f"({ev_start.strftime('%I:%M %p')}–{ev_end.strftime('%I:%M %p')}) on {date}."
            )

    # ── Create the event ──────────────────────────────────────────
    event_body = {
        'summary': title,
        'start': {
            'dateTime': new_start.isoformat(),
            'timeZone': 'Asia/Kolkata'
        },
        'end': {
            'dateTime': new_end.isoformat(),
            'timeZone': 'Asia/Kolkata'
        },
    }

    if attendee_email:
        event_body['attendees'] = [{'email': attendee_email}]

    created = service.events().insert(
        calendarId='primary',
        body=event_body,
        sendUpdates='all' if attendee_email else 'none'
    ).execute()

    link = created.get('htmlLink', 'No link returned')
    return (
        f"✅ Event '{title}' created successfully on {date} at {start_time} "
        f"for {duration_minutes} minutes.\nLink: {link}"
    )


# ─────────────────────────────────────────────
# TOOL: find_free_slots  (Task 4)
# ─────────────────────────────────────────────
@tool
def find_free_slots(date: str, duration_minutes: int) -> str:
    """
    Find all available time slots on a given date that can fit a meeting
    of the specified duration, within working hours (9 AM to 6 PM IST).

    Args:
        date: YYYY-MM-DD
        duration_minutes: Required meeting duration in minutes

    Returns:
        A readable list of free slots, or a message if none available.
    """
    service = get_calendar_service()
    day = datetime.date.fromisoformat(date)

    work_start = datetime.datetime(day.year, day.month, day.day, WORK_START, 0, tzinfo=TZ)
    work_end   = datetime.datetime(day.year, day.month, day.day, WORK_END,   0, tzinfo=TZ)

    events_result = service.events().list(
        calendarId='primary',
        timeMin=work_start.isoformat(),
        timeMax=work_end.isoformat(),
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    busy_blocks = []
    for ev in events_result.get('items', []):
        ev_start = parse_event_time(ev['start'].get('dateTime', ev['start'].get('date')))
        ev_end   = parse_event_time(ev['end'].get('dateTime',   ev['end'].get('date')))
        # Clamp to working hours
        ev_start = max(ev_start, work_start)
        ev_end   = min(ev_end,   work_end)
        if ev_start < ev_end:
            busy_blocks.append((ev_start, ev_end))

    # Sort and merge overlapping busy blocks
    busy_blocks.sort(key=lambda x: x[0])
    merged = []
    for start, end in busy_blocks:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append([start, end])

    # Find gaps
    free_slots = []
    cursor = work_start
    for busy_start, busy_end in merged:
        if (busy_start - cursor).total_seconds() >= duration_minutes * 60:
            free_slots.append((cursor, busy_start))
        cursor = max(cursor, busy_end)
    if (work_end - cursor).total_seconds() >= duration_minutes * 60:
        free_slots.append((cursor, work_end))

    if not free_slots:
        return (
            f"No free slots on {date} for a {duration_minutes}-minute meeting "
            f"during working hours (9 AM–6 PM)."
        )

    lines = [f"Free slots on {date} (for a {duration_minutes}-minute meeting):"]
    for s, e in free_slots:
        lines.append(f"  • {s.strftime('%I:%M %p')} → {e.strftime('%I:%M %p')}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOL: analyse_booking_patterns  (Task 4)
# ─────────────────────────────────────────────
@tool
def analyse_booking_patterns() -> str:
    """
    Analyse the user's Google Calendar for the past 30 days to identify
    scheduling habits: busiest days, lightest days, preferred meeting hours,
    and average meeting duration.

    Returns:
        A structured summary of booking patterns.
    """
    service = get_calendar_service()
    now = datetime.datetime.now(tz=TZ)
    past_30 = now - datetime.timedelta(days=30)

    events_result = service.events().list(
        calendarId='primary',
        timeMin=past_30.isoformat(),
        timeMax=now.isoformat(),
        maxResults=200,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    items = events_result.get('items', [])

    if not items:
        return "No events found in the past 30 days to analyse."

    day_counts  = {}   # weekday_name → count
    hour_counts = {}   # hour (int) → count
    durations   = []   # list of duration in minutes

    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for ev in items:
        start_raw = ev['start'].get('dateTime', ev['start'].get('date'))
        end_raw   = ev['end'].get('dateTime',   ev['end'].get('date'))
        ev_start  = parse_event_time(start_raw)
        ev_end    = parse_event_time(end_raw)

        day_name = DAYS[ev_start.weekday()]
        day_counts[day_name] = day_counts.get(day_name, 0) + 1

        hour = ev_start.hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1

        dur = (ev_end - ev_start).total_seconds() / 60
        if 0 < dur <= 480:
            durations.append(dur)

    total_events = len(items)
    avg_duration = round(sum(durations) / len(durations)) if durations else 0

    sorted_days  = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)

    busiest_day  = sorted_days[0][0]  if sorted_days              else "N/A"
    lightest_day = sorted_days[-1][0] if len(sorted_days) > 1     else "N/A"
    peak_hour    = f"{sorted_hours[0][0]:02d}:00" if sorted_hours else "N/A"

    lines = [
        f"📊 Booking Pattern Analysis (last 30 days):",
        f"  • Total meetings   : {total_events}",
        f"  • Busiest day      : {busiest_day} ({day_counts.get(busiest_day, 0)} meetings)",
        f"  • Lightest day     : {lightest_day} ({day_counts.get(lightest_day, 0)} meetings)",
        f"  • Peak hour        : {peak_hour}",
        f"  • Avg duration     : {avg_duration} minutes",
        f"",
        f"  Day breakdown:",
    ]
    for day, count in sorted_days:
        lines.append(f"    {day}: {count} meeting(s)")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOL: query_calendar_insights  (Bonus)
# ─────────────────────────────────────────────
@tool
def query_calendar_insights(question: str) -> str:
    """
    Answer general natural language questions about the user's calendar.

    Handles questions like:
      - "Which days am I free this week?"
      - "What was my busiest day this month?"
      - "How many hours of meetings do I have this week?"
      - "What meetings do I have tomorrow?"

    Args:
        question: A natural language question about the user's calendar.

    Returns:
        A natural language answer based on real calendar data.
    """
    service = get_calendar_service()
    now = datetime.datetime.now(tz=TZ)

    q = question.lower()
    if "this week" in q:
        start_of_period = now - datetime.timedelta(days=now.weekday())
        start_of_period = start_of_period.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_period   = start_of_period + datetime.timedelta(days=7)
        label = "this week"
    elif "this month" in q:
        start_of_period = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_of_period   = (start_of_period + datetime.timedelta(days=32)).replace(day=1)
        label = "this month"
    elif "tomorrow" in q:
        tomorrow = now + datetime.timedelta(days=1)
        start_of_period = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_period   = start_of_period + datetime.timedelta(days=1)
        label = "tomorrow"
    elif "today" in q:
        start_of_period = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_period   = start_of_period + datetime.timedelta(days=1)
        label = "today"
    elif "last 30" in q or "past 30" in q:
        start_of_period = now - datetime.timedelta(days=30)
        end_of_period   = now
        label = "the past 30 days"
    else:
        start_of_period = now - datetime.timedelta(days=now.weekday())
        start_of_period = start_of_period.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_period   = start_of_period + datetime.timedelta(days=7)
        label = "this week"

    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_of_period.isoformat(),
        timeMax=end_of_period.isoformat(),
        maxResults=200,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    items = events_result.get('items', [])

    if not items:
        return f"You have no events scheduled for {label}."

    total_minutes = 0
    by_day = {}
    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for ev in items:
        start_raw = ev['start'].get('dateTime', ev['start'].get('date'))
        end_raw   = ev['end'].get('dateTime',   ev['end'].get('date'))
        ev_start  = parse_event_time(start_raw)
        ev_end    = parse_event_time(end_raw)
        dur_min   = (ev_end - ev_start).total_seconds() / 60

        date_key = ev_start.strftime('%Y-%m-%d')
        if date_key not in by_day:
            by_day[date_key] = []
        by_day[date_key].append({
            'name':     ev.get('summary', 'Untitled'),
            'start':    ev_start,
            'end':      ev_end,
            'duration': dur_min
        })
        if 0 < dur_min <= 480:
            total_minutes += dur_min

    total_hours   = round(total_minutes / 60, 1)
    busiest_date  = max(by_day, key=lambda d: len(by_day[d]))
    lightest_date = min(by_day, key=lambda d: len(by_day[d]))
    busiest_day_name  = DAYS[datetime.date.fromisoformat(busiest_date).weekday()]
    lightest_day_name = DAYS[datetime.date.fromisoformat(lightest_date).weekday()]

    all_days_in_range = []
    cursor = start_of_period.date()
    while cursor < end_of_period.date():
        all_days_in_range.append(cursor.isoformat())
        cursor += datetime.timedelta(days=1)
    free_days      = [d for d in all_days_in_range if d not in by_day]
    free_day_names = [DAYS[datetime.date.fromisoformat(d).weekday()] for d in free_days]

    lines = [f"📅 Calendar Insights for {label}:"]
    lines.append(f"  • Total meetings   : {len(items)}")
    lines.append(f"  • Total hours      : {total_hours} hrs")
    lines.append(f"  • Busiest day      : {busiest_day_name} ({busiest_date}) — {len(by_day[busiest_date])} meetings")
    lines.append(f"  • Lightest day     : {lightest_day_name} ({lightest_date}) — {len(by_day[lightest_date])} meeting(s)")

    if free_days:
        lines.append(f"  • Free days        : {', '.join(free_day_names)} ({', '.join(free_days)})")
    else:
        lines.append(f"  • Free days        : None — every day has at least one meeting!")

    lines.append("")
    lines.append("  Day-by-day breakdown:")
    for date_str in sorted(by_day.keys()):
        day_name = DAYS[datetime.date.fromisoformat(date_str).weekday()]
        evs = by_day[date_str]
        lines.append(f"    {day_name} {date_str}: {len(evs)} meeting(s)")
        for e in evs:
            lines.append(
                f"      – {e['name']} @ {e['start'].strftime('%I:%M %p')} ({int(e['duration'])} min)"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# TOOL: extract_meeting_from_image  (Multi-Modal)
# ─────────────────────────────────────────────
@tool
def extract_meeting_from_image(image_path: str) -> str:
    """
    Extract meeting details from an image (e.g., a screenshot of a WhatsApp message,
    email invite, or calendar screenshot) using Gemini's vision capabilities.

    Accepts JPEG, PNG, GIF, and WEBP images. Returns a structured summary of the
    extracted meeting details: title, date, time, duration, and attendee email (if visible).

    After calling this tool, use create_event() with the extracted fields to schedule it.

    Args:
        image_path: Absolute or relative path to the image file on disk.

    Returns:
        A JSON-like string with extracted fields, or an error message if extraction fails.
    """
    # ── Validate file ─────────────────────────────────────────────
    if not os.path.exists(image_path):
        return f"Error: File not found at '{image_path}'. Please provide a valid image path."

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext)
    if not mime_type:
        return f"Error: Unsupported image format '{ext}'. Use JPEG, PNG, GIF, or WEBP."

    # ── Read and encode image ─────────────────────────────────────
    try:
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"Error reading image file: {e}"

    # ── Call Gemini Vision ─────────────────────────────────────────
    today = datetime.date.today().isoformat()
    vision_prompt = f"""You are a meeting detail extractor. Today's date is {today}.

Carefully examine this image (which may be a WhatsApp screenshot, email, calendar invite,
or any other message containing meeting information) and extract:

1. **title** – the meeting/event name (use "Meeting" if unclear)
2. **date** – in YYYY-MM-DD format (compute from relative phrases like "tomorrow", "next Monday")
3. **start_time** – in HH:MM 24-hour format (e.g., 14:30 for 2:30 PM)
4. **duration_minutes** – integer (default to 60 if not specified)
5. **attendee_email** – email address if visible, else null

Return ONLY a JSON object with exactly these keys, no extra text:
{{
  "title": "...",
  "date": "YYYY-MM-DD",
  "start_time": "HH:MM",
  "duration_minutes": 60,
  "attendee_email": null
}}

If no meeting information is found in the image, return:
{{"error": "No meeting information found in image."}}"""

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
        response = llm.invoke([
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
                {
                    "type": "text",
                    "text": vision_prompt,
                },
            ])
        ])

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()

        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"):
            content = "\n".join(content.split("\n")[:-1])
        content = content.strip()

        return f"Extracted meeting details from image:\n{content}"

    except Exception as e:
        return f"Error calling Gemini Vision: {e}"