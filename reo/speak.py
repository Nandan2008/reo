import pyttsx3


def speak(text: str) -> None:
    print(f"Reo: {text}")
    # A fresh engine per call avoids a known pyttsx3 hang on Windows
    engine = pyttsx3.init()
    engine.setProperty("rate", 185)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
