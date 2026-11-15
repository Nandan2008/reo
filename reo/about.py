"""What reo knows about itself.

Asked these, reo used to guess, and guessing about yourself goes wrong in a
particular way: it said it was made by OpenAI three times running and kept
saying it after being corrected, and it told him to say "stop" to end the
program, which is not one of the words that works.

None of this needs a model. The name is in config, the words that stop it
are in __main__, the tools are a list, the folder is a path, and which model
is answering is whatever last answered. So these are assembled from what is
actually true at the time and never leave the machine - which also means
they still work when the allowance is spent, which is exactly when someone
is most likely to be asking what is going on.
"""
from . import config


def _stopping() -> str:
    from .__main__ import EXIT_PHRASES
    words = sorted(p for p in EXIT_PHRASES if " " not in p)
    return (f"Say {', '.join(words[:-1])} or {words[-1]}. "
            "Ctrl+Q closes the window, and Escape takes it out of full screen.")


def _who_built_it() -> str:
    return ("You did. I run entirely on this laptop - the code is in your reo "
            "folder, and you can change any of it.")


def _what_it_is() -> str:
    from .hotkey import describe
    how = {"hotkey": f"press {describe()}",
           "wake": "say the wake word",
           "push": "press Enter in my window"}.get(config.MODE, "press the key")
    return (f"I am Reo, a voice assistant running on this laptop. To talk to "
            f"me, {how}.")


def _what_it_can_do() -> str:
    return ("I can open programs and websites, play a video, search the web "
            "and read a page, write files and run them, control the volume, "
            "the music and the windows, move the mouse where you tell me, "
            "remember things about you, and put the last thing back if I got "
            "it wrong. Say undo that.")


def _which_model() -> str:
    from . import providers
    thinking = providers.active_label()
    ear = ("Groq's big whisper" if config.HEARING == "hosted"
           else f"the {config.WHISPER_MODEL} speech model here")
    return f"I think with {thinking}, and I hear you with {ear}."


def _where_files_go() -> str:
    return f"Everything I write goes in {config.WORKSPACE}."


def _what_it_hears_with() -> str:
    return _which_model()


# Whole-utterance patterns, the same rule as everything else in fastpath.
ABOUT = [
    (r"(who|what) (made|created|built|wrote|designed) you"
     r"|who( is|s) your (maker|creator|developer)"
     r"|(who|what) are you (made|created|built) by", _who_built_it),
    (r"(who|what) are you|what are you called|whats your name"
     r"|introduce yourself|tell me about yourself", _what_it_is),
    (r"(what|which) (can|could) you do|what are you (able to do|capable of)"
     r"|what can you help( me)? with|what do you do"
     r"|(what|which) things can you do", _what_it_can_do),
    (r"how do i (stop|close|quit|exit|end) (you|this|the (program|app))"
     r"|how do i turn you off|how do i shut you (down|up)"
     r"|how do i end the program|how do i get rid of you", _stopping),
    (r"(what|which) (model|brain|ai) (are you|do you) (using|use|run on|running)"
     r"|what are you (built|running) on|what model are you"
     r"|(what|which) model is (this|that)", _which_model),
    (r"where do you (put|save|write|keep) (the |my |your )?(files|code|things|stuff)"
     r"|where (do|does) (the |my |your )?(files|code) go"
     r"|wheres your folder|where is your folder", _where_files_go),
]
