from __future__ import annotations

from threading import Thread

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None


def speak_text(text: str, rate: int = 170) -> None:
    if pyttsx3 is None:
        print(f"[TTS] {text}")
        return
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()


def speak_async(text: str, rate: int = 170) -> Thread:
    thread = Thread(target=speak_text, args=(text, rate), daemon=True)
    thread.start()
    return thread
