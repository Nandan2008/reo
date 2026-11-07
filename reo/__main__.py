import sys
import time
from datetime import datetime
from pathlib import Path

from . import config, fastpath
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


def already_running() -> int:
    """PID of another reo, or 0. Two of them fight over the microphone and the
    speaker, and the loser waits forever on a sound device it cannot have."""
    import os

    try:
        import psutil
    except ImportError:
        return 0
    # The venv's python.exe launches a child with the same command line, so the
    # launcher looks exactly like a second copy. Skip the whole ancestor chain.
    family = {os.getpid()}
    try:
        walker = psutil.Process().parent()
        while walker is not None and len(family) < 8:
            family.add(walker.pid)
            walker = walker.parent()
    except Exception:
        pass

    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = process.info
            pid, name, cmdline = info["pid"], info["name"] or "", info["cmdline"] or []
        except Exception:
            continue
        if pid in family or not name.lower().startswith("python"):
            continue
        # "-m" immediately followed by "reo", and nothing looser. Matching any
        # part ending in "reo" caught the working directory and refused to start
        # when nothing was running at all.
        if any(a == "-m" and b == "reo" for a, b in zip(cmdline, cmdline[1:])):
            return pid
    return 0


def woken_by() -> str:
    """What to tell the user to say, which depends on the wake engine."""
    if config.WAKE_ENGINE == "phrase":
        from .phrase import pretty
        return pretty(config.WAKE_PHRASE)
    return config.WAKE_NAME


def handle(brain, text: str, ui=None, heard_in: float = 0.0) -> bool:
    """Answer one command. Returns False when it is time to stop."""
    print(f"You: {text}")
    if ui:
        ui.line("you", text)

    if normalize(text) in EXIT_PHRASES:
        speak("Goodbye.", ui)
        return False

    # Things the machine already knows need no model: answering the time took
    # two seconds of round trips before this.
    quick = fastpath.answer(text)
    if quick is not None:
        if ui:
            ui.set_state("thinking")
        started = time.time()
        speak(quick, ui)
        # keep the exchange in history, or a follow-up loses the thread
        brain.history.append({"role": "user", "content": text})
        brain.history.append({"role": "assistant", "content": quick})
        print(f"  [time] heard {heard_in:.1f}s  answered directly (no model)  "
              f"spoke {time.time() - started:.1f}s")
        return True

    if ui:
        ui.set_state("thinking")

    started = time.time()
    try:
        reply = brain.think(text)
    except Exception as e:
        print(f"  [error] {e}")
        reply = "Something went wrong reaching my brain. Please try again."
    thought = time.time() - started

    started = time.time()
    speak(reply or "Done.", ui)
    spoke = time.time() - started

    # Where a reply's time actually went, so slowness can be read off the log
    # instead of guessed at.
    calls = sum(1 for m in brain.history if m.get("role") == "tool")
    print(f"  [time] heard {heard_in:.1f}s  thought {thought:.1f}s ({calls} tools)  "
          f"spoke {spoke:.1f}s  total {heard_in + thought + spoke:.1f}s")
    return True


def main():
    # Windows consoles are cp1252; replies often hold characters it cannot encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _tee_to_logfile()

    if not config.LLM_API_KEY or config.LLM_API_KEY == "your-groq-key-here":
        print("Missing LLM_API_KEY. Copy .env.example to .env and add your free Groq key.")
        return

    other = already_running()
    if other:
        print(f"reo is already running (process {other}). Close that one first.")
        print("Two copies fight over the microphone and the speaker, which looks")
        print("exactly like reo being slow or frozen.")
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
    if config.UI != "off":
        if config.UI == "tk":
            from .ui import Display
        else:
            try:
                from .webui import Display
            except Exception as e:
                print(f"  [ui] the web window is unavailable ({e}); using the plain one.")
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
        heard_in = 0.0
        try:
            if ui:
                ui.set_state("idle")
            if toggle:
                if not toggle.listening:
                    print(f"\n[Press {combination} to start listening]")
                    toggle.wait_until_listening()
                if ui:
                    ui.set_state("listening")
                started = time.time()
                text = listener.listen()
                heard_in = time.time() - started
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

        if not handle(brain, text, ui, heard_in):
            break


if __name__ == "__main__":
    main()
