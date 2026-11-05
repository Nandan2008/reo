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


def woken_by() -> str:
    """What to tell the user to say, which depends on the wake engine."""
    if config.WAKE_ENGINE == "phrase":
        return config.WAKE_PHRASE.replace("_", " ")
    return config.WAKE_NAME


def handle(brain, text: str, ui=None) -> bool:
    """Answer one command. Returns False when it is time to stop."""
    print(f"You: {text}")
    if ui:
        ui.line("you", text)

    if normalize(text) in EXIT_PHRASES:
        speak("Goodbye.", ui)
        return False

    if ui:
        ui.set_state("thinking")
    try:
        reply = brain.think(text)
    except Exception as e:
        print(f"  [error] {e}")
        reply = "Something went wrong reaching my brain. Please try again."
    speak(reply or "Done.", ui)
    return True


def main():
    # Windows consoles are cp1252; replies often hold characters it cannot encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _tee_to_logfile()

    if not config.LLM_API_KEY or config.LLM_API_KEY == "your-groq-key-here":
        print("Missing LLM_API_KEY. Copy .env.example to .env and add your free Groq key.")
        return

    listener, brain = Listener(), Brain()

    waker = toggle = combination = None
    if config.MODE == "wake":
        if config.WAKE_ENGINE == "phrase":
            from .phrase import PhraseListener
            waker = PhraseListener(listener)
        else:
            from .wake import WakeListener
            waker = WakeListener()
    elif config.MODE == "hotkey":
        from .hotkey import Toggle, describe
        combination = describe()
        toggle = Toggle(on_change=lambda on: print(
            f"\n[{combination}] {'listening' if on else 'stopped listening'}"))
        toggle.start()

    ui = None
    if config.UI:
        from .ui import Display
        hint = {"hotkey": f"{combination} to listen",
                "wake": f'say "{woken_by()}"',
                "push": "press Enter in the console"}.get(config.MODE, "")
        ui = Display(subtitle=hint)
        ui.run(lambda: _loop(listener, brain, waker, toggle, combination, ui))
        return

    _loop(listener, brain, waker, toggle, combination, None)


def _loop(listener, brain, waker, toggle, combination, ui):
    speak("Reo online.", ui)

    while not (ui and ui.closed):
        try:
            if ui:
                ui.set_state("idle")
            if toggle:
                if not toggle.listening:
                    print(f"\n[Press {combination} to start listening]")
                    toggle.wait_until_listening()
                if ui:
                    ui.set_state("listening")
                text = listener.listen()
                if not toggle.listening:
                    continue          # switched off while it was recording
            elif waker:
                print(f'\n[Listening for "{woken_by()}"]')
                text = waker.wait()
                if not is_speech(text):
                    # woken with nothing after it, so ask what they want
                    speak("[happy] Yes?", ui)
                    if ui:
                        ui.set_state("listening")
                    text = listener.listen()
            else:
                input("\n[Press Enter, then speak]  ")
                print("Listening...")
                if ui:
                    ui.set_state("listening")
                text = listener.listen()
        except (KeyboardInterrupt, EOFError):
            break

        if not is_speech(text):
            print("(didn't catch that)")
            if ui:
                ui.line("note", "(didn't catch that)")
            continue

        if not handle(brain, text, ui):
            break


if __name__ == "__main__":
    main()
