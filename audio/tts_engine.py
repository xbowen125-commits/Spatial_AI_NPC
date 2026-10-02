"""基于 Windows SAPI / pyttsx3 的轻量本地 TTS。"""

from dataclasses import dataclass

from .audio_config import TTS_RATE, TTS_VOICE_ID, TTS_VOLUME


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    name: str
    languages: tuple[str, ...]


class LocalTTSEngine:
    """只在 TTS 工作线程中创建和使用 pyttsx3 Engine。"""

    def __init__(self):
        try:
            import pyttsx3
        except ImportError as error:
            raise RuntimeError(
                "未安装 pyttsx3，请先运行：python -m pip install pyttsx3"
            ) from error

        try:
            self._engine = pyttsx3.init("sapi5")
        except Exception as error:
            raise RuntimeError(f"Windows SAPI TTS 初始化失败：{error}") from error

        self._engine.setProperty("rate", TTS_RATE)
        self._engine.setProperty("volume", max(0.0, min(1.0, TTS_VOLUME)))
        self._voices = list(self._engine.getProperty("voices") or [])
        self._default_voice_id = self._engine.getProperty("voice")
        self._warned_no_chinese = False

    @property
    def engine_name(self):
        return "pyttsx3 / Windows SAPI5"

    @property
    def rate(self):
        return self._engine.getProperty("rate")

    @property
    def volume(self):
        return self._engine.getProperty("volume")

    @property
    def current_voice(self):
        return self._engine.getProperty("voice")

    def list_voices(self):
        return [
            VoiceInfo(
                id=str(voice.id),
                name=str(getattr(voice, "name", "Unknown")),
                languages=tuple(
                    self._language_text(value)
                    for value in (getattr(voice, "languages", None) or [])
                ),
            )
            for voice in self._voices
        ]

    def find_voice(self, language):
        """优先显式 Voice ID，否则按系统 Voice 名称与语言标签选择。"""
        if TTS_VOICE_ID:
            for voice in self._voices:
                if str(voice.id) == str(TTS_VOICE_ID):
                    return voice

        is_chinese = str(language).lower().startswith("zh")
        tokens = (
            ("zh", "chinese", "中文", "huihui", "yaoyao", "xiaoxiao", "yunxi")
            if is_chinese
            else ("en", "english", "zira", "david", "mark")
        )
        for voice in self._voices:
            searchable = self._voice_search_text(voice)
            if any(token in searchable for token in tokens):
                return voice
        return None

    def speak(self, text, language="unknown"):
        text = str(text).strip()
        if not text:
            raise ValueError("TTS 文本为空。")

        voice = self.find_voice(language)
        if voice is not None:
            self._engine.setProperty("voice", voice.id)
        elif str(language).lower().startswith("zh"):
            self._engine.setProperty("voice", self._default_voice_id)
            if not self._warned_no_chinese:
                print("No Chinese system voice found. 使用系统默认 Voice。")
                self._warned_no_chinese = True
        else:
            self._engine.setProperty("voice", self._default_voice_id)

        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self):
        try:
            self._engine.stop()
        except Exception:
            pass

    @staticmethod
    def _language_text(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore") or repr(value)
        return str(value)

    def _voice_search_text(self, voice):
        values = [str(voice.id), str(getattr(voice, "name", ""))]
        values.extend(
            self._language_text(value)
            for value in (getattr(voice, "languages", None) or [])
        )
        return " ".join(values).lower()
