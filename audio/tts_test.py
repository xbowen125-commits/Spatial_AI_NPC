"""列出 Windows Voice，并测试中文与英文 TTS。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.tts_engine import LocalTTSEngine


def main():
    try:
        engine = LocalTTSEngine()
        voices = engine.list_voices()
        print(f"Engine: {engine.engine_name}")
        print(f"Current Voice: {engine.current_voice}")
        print(f"Rate: {engine.rate}")
        print(f"Volume: {engine.volume}")
        print("Available Voices:")
        for index, voice in enumerate(voices):
            print(f"  [{index}] {voice.name} | {voice.id} | {voice.languages}")

        chinese_voice = engine.find_voice("zh")
        if chinese_voice is None:
            print("No Chinese system voice found.")
        else:
            print(f"Chinese Voice: {getattr(chinese_voice, 'name', chinese_voice.id)}")
            engine.speak("你好，我现在可以说话了。", "zh")

        english_voice = engine.find_voice("en")
        if english_voice is not None:
            print(f"English Voice: {getattr(english_voice, 'name', english_voice.id)}")
        engine.speak("Hello, I can speak now.", "en")
    except Exception as error:
        print(f"TTS 测试失败：{error}")


if __name__ == "__main__":
    main()
