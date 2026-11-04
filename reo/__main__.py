import sys

from . import config
from .brain import Brain
from .listen import Listener
from .speak import speak

EXIT_PHRASES = {"goodbye", "bye", "exit", "quit", "stop listening", "goodbye reo", "bye reo"}


def is_speech(text: str) -> bool:
    """Whisper emits bare punctuation on room noise; that is not worth an API call."""
    return any(c.isalnum() for c in text)


def normalize(text: str) -> str:
    """Strip Whisper's punctuation and spacing so phrases match."""
    kept = "".join(c if c.isalnum() else " " for c in text.lower())
    return " ".join(kept.split())


def main():
    # Windows consoles are cp1252; replies often hold characters it cannot encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not config.LLM_API_KEY or config.LLM_API_KEY == "your-groq-key-here":
        print("Missing LLM_API_KEY. Copy .env.example to .env and add your free Groq key.")
        return

    listener, brain = Listener(), Brain()
    speak("Reo online.")

    while True:
        try:
            input("\n[Press Enter, then speak]  ")
        except (KeyboardInterrupt, EOFError):
            break
        print("Listening...")
        text = listener.listen()
        if not is_speech(text):
            print("(didn't catch that)")
            continue
        print(f"You: {text}")

        if normalize(text) in EXIT_PHRASES:
            speak("Goodbye.")
            break

        try:
            reply = brain.think(text)
        except Exception as e:
            print(f"  [error] {e}")
            reply = "Something went wrong reaching my brain. Please try again."
        speak(reply or "Done.")


if __name__ == "__main__":
    main()
