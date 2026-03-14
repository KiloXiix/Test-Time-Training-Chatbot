"""
main.py — Entry point for the Companion AI.

Run with:  python main.py
"""

import sys
import os

# Ensure companion_ai directory is on path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from companion import CompanionAI, RECOMMENDED_MODELS
from database import initialize_database, get_connection


# ─────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Companion AI — Learns As You Talk                  ║
║                                                              ║
║  Like a baby learning language:                              ║
║   • Remembers what you tell it across sessions               ║
║   • Learns new words and their meanings                      ║
║   • Accepts corrections and updates its beliefs              ║
║   • Adapts its language patterns to match yours              ║
║                                                              ║
║  Powered by: TTT + Belief Store + Vocabulary Module          ║
╚══════════════════════════════════════════════════════════════╝
""")


def select_model() -> str:
    print("  Select a model:\n")
    for key, (model_id, desc) in RECOMMENDED_MODELS.items():
        print(f"    [{key}] {desc}")
        print(f"        {model_id}")
    print()
    choice = input("  Choice (1-5, or paste a HuggingFace model ID): ").strip()

    if choice in RECOMMENDED_MODELS:
        model_id = RECOMMENDED_MODELS[choice][0]
    elif "/" in choice:
        model_id = choice
    else:
        print("  Invalid choice, using default.")
        model_id = RECOMMENDED_MODELS["1"][0]

    print(f"\n  Using: {model_id}\n")
    return model_id


def select_user(model_id: str) -> str:
    initialize_database()
    conn = get_connection()

    rows = conn.execute("""
        SELECT DISTINCT cl.user_id,
               COUNT(*) as messages,
               MAX(cl.timestamp) as last_seen
        FROM conversation_log cl
        GROUP BY cl.user_id
        ORDER BY last_seen DESC
        LIMIT 10
    """).fetchall()
    conn.close()

    if rows:
        print("  Existing companions:\n")
        for r in rows:
            import datetime
            ts = datetime.datetime.fromtimestamp(r["last_seen"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            print(f"    • {r['user_id']}  ({r['messages']} messages, last: {ts})")
        print()

    user_id = input(
        "  Your name / user ID (Enter = 'friend'): "
    ).strip() or "friend"

    print(f"\n  Hello, {user_id}!\n")
    return user_id


def select_ai_name() -> str:
    """Ask the user what to call the AI companion."""
    ai_name = input(
        "  AI companion name (Enter = 'Companion'): "
    ).strip() or "Companion"
    print(f"\n  Your companion's name is: {ai_name}\n")
    return ai_name


def print_help():
    print("""
  Commands:
    /verbose    — Toggle detailed learning + TTT stats each turn
    /stats      — Show full learning statistics
    /beliefs    — Show everything the AI believes about you
    /vocab      — Show vocabulary acquisition status
    /save       — Force-save the current session
    /clear      — Clear screen
    /help       — Show this help
    /quit       — Save and exit
""")


# ─────────────────────────────────────────────────────────────
# KNOWN COMMANDS — used to catch near-typos before chat
# ─────────────────────────────────────────────────────────────

COMMANDS = {"/verbose", "/stats", "/beliefs", "/vocab",
            "/save", "/clear", "/help", "/quit"}


def looks_like_command(text: str) -> bool:
    """Return True if the input starts with / and looks like a mistyped command."""
    return text.startswith("/")


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main():
    print_banner()

    model_id = select_model()
    user_id  = select_user(model_id)
    ai_name  = select_ai_name()

    print("  Loading companion (first run downloads model weights)...\n")
    companion = CompanionAI(user_id=user_id, model_id=model_id, ai_name=ai_name)

    verbose = False
    print_help()
    print("─" * 64)
    print("  Start talking! The more you talk, the more it learns.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saving and exiting...")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        # ── Commands ─────────────────────────────────────────────────
        if cmd == "/quit":
            break
        elif cmd == "/verbose":
            verbose = not verbose
            print(f"  Verbose mode: {'ON' if verbose else 'OFF'}\n")
            continue
        elif cmd == "/stats":
            companion.stats()
            continue
        elif cmd == "/beliefs":
            companion.show_beliefs()
            continue
        elif cmd == "/vocab":
            companion.show_vocabulary()
            continue
        elif cmd == "/save":
            companion._save_session()
            print()
            continue
        elif cmd == "/help":
            print_help()
            continue
        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        elif looks_like_command(user_input):
            # Catch typos like /quir, /quite, /stats2, etc.
            # Don't pass them through to the model as chat messages.
            print(f"  Unknown command '{user_input}'. Type /help for a list.\n")
            continue

        # ── Normal conversation turn ──────────────────────────────────
        companion.chat(user_input, verbose=verbose)
        print()

    # Final save on exit
    companion._save_session()
    companion.stats()
    print("  Session saved. Goodbye!\n")


if __name__ == "__main__":
    main()