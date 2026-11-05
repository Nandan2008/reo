import sys
from datetime import datetime
from pathlib import Path

from . import config
from .brain import Brain
from .listen import Listener
from .speak import speak
from .text import is_speech, normalize

EXIT_PHRASES = {"goodbye", "bye", "exit", "quit", "stop listening", "goodbye reo", "bye reo"}


def _tee_to_logfile():
    """Started from the Startup folder there is no console to read, so keep a log."""
    path = Path(__file__).resolve().parent.parent / "reo.log"
    try:
        log = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        return
    log.write(f"\n===== started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")

    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            log.write(text)
            return len(text)

        def flush(self):
            sys.__stdout__.flush()
            log.flush()

    sys.stdout = Tee()


def handle(brain, text: str) -> bool:
    """Answer one command. Returns False when it is time to stop."""
    print(f"You: {text}")

    if normalize(text) in EXIT_PHRASES:
        speak("Goodbye.")
        return False

    try:
        reply = brain.think(text)
    except Exception as e:
        print(f"  [error] {e}")
        reply = "Something went wrong reaching my brain. Please try again."
    speak(reply or "Done.")
    return True


def main():
    # Windows consoles are cp1252; replies often hold characters it cannot encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _tee_to_logfile()

    if not config.LLM_API_KEY or config.LLM_API_KEY == "your-groq-key-here":
        print("Missing LLM_API_KEY. Copy .env.example to .env and add your free Groq key.")
        return

    listener, brain = Listener(), Brain()

    waker = None
    if config.MODE == "wake":
        from .wake import WakeListener
        waker = WakeListener()

    speak("Reo online.")

    while True:
        try:
            if waker:
                print(f'\n[Listening for "{config.WAKE_NAME}"]')
                text = waker.wait()
                if not is_speech(text):
                    # addressed by name with nothing after it
                    speak("[happy] Yes?")
                    text = listener.listen()
            else:
                input("\n[Press Enter, then speak]  ")
                print("Listening...")
                text = listener.listen()
        except (KeyboardInterrupt, EOFError):
            break

        if not is_speech(text):
            print("(didn't catch that)")
            continue

        if not handle(brain, text):
            break


if __name__ == "__main__":
    main()
