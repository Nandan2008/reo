from . import config
from .brain import Brain
from .listen import Listener
from .speak import speak

EXIT_PHRASES = {"goodbye", "bye", "exit", "quit", "stop listening", "goodbye reo", "bye reo"}


def main():
    if not config.LLM_API_KEY:
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
        if not text:
            print("(didn't catch that)")
            continue
        print(f"You: {text}")

        if text.lower().strip(" .!?,") in EXIT_PHRASES:
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
