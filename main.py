from agent import create_scheduler_agent
import datetime

def print_banner():
    print("=" * 60)
    print("  🗓️  AI Meeting Scheduler — Powered by Gemini + LangChain")
    print("=" * 60)
    print(f"  Today: {datetime.date.today()}  |  Type 'exit' to quit")
    print("=" * 60)
    print()
    print("  Example prompts:")
    print("  • Schedule a 1-hour meeting called Team Sync tomorrow at 10am")
    print("  • Book a 30-min standup at 9am this Friday")
    print("  • Set up a 45-min call with raj@example.com on Monday at 3pm")
    print("  • Am I free next Tuesday at 2pm?")
    print("  • How many hours of meetings do I have this week?")
    print("  • Which days am I free this week?")
    print()
    print("  📸 Multi-Modal — schedule from a screenshot:")
    print("  • image /path/to/invite.png")
    print("  • image /path/to/whatsapp.jpg Schedule this for me")
    print()
    print("  🧠 RAG context from your past meetings is used automatically.")
    print()


def parse_image_input(user_input: str):
    """
    Detect if the user typed 'image <path> [optional text]'.
    Returns (image_path, extra_text) or (None, None) if not an image command.
    """
    parts = user_input.split(maxsplit=2)
    if len(parts) < 2:
        return None, None

    image_path = parts[1]
    extra_text = parts[2] if len(parts) > 2 else ""
    return image_path, extra_text


def main():
    print_banner()

    try:
        agent = create_scheduler_agent()
        print("[✓] Agent initialized successfully.\n")
    except Exception as e:
        print(f"[✗] Failed to initialize agent: {e}")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ('exit', 'quit', 'bye'):
            print("Goodbye! 👋")
            break

        # ── Image input detection ──────────────────────────────────
        if user_input.lower().startswith("image "):
            image_path, extra_text = parse_image_input(user_input)
            if image_path:
                # Build a prompt that tells the agent to use the image tool
                if extra_text:
                    prompt = (
                        f"The user has provided an image at path '{image_path}'. "
                        f"First call extract_meeting_from_image with this path to read the meeting details, "
                        f"then act on this additional instruction: {extra_text}"
                    )
                else:
                    prompt = (
                        f"The user has provided an image at path '{image_path}'. "
                        f"Call extract_meeting_from_image with this path to extract the meeting details, "
                        f"then schedule the meeting by calling create_event with the extracted fields."
                    )
                user_input = prompt

        print()
        try:
            result = agent(user_input)
            print(f"Assistant: {result}")
        except Exception as e:
            print(f"[Error] {e}")
        print()


if __name__ == "__main__":
    main()