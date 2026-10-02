"""不依赖 LLM 的可替换规则回复引擎。"""

from audio.audio_config import RESPOND_TO_UNKNOWN_DIRECTED_SPEECH
from events.interaction_event import ResponseEvent


class RuleBasedResponseEngine:
    def create_response(self, speech_event):
        context = speech_event.context
        directed = context.directed_speech
        if directed == "false":
            return None
        if directed == "unknown" and not RESPOND_TO_UNKNOWN_DIRECTED_SPEECH:
            return None

        text = speech_event.text.strip().lower()
        use_chinese = str(speech_event.language).lower().startswith("zh")

        if (
            self._contains(text, ("再见", "拜拜"))
            or text in {"goodbye", "bye", "bye bye"}
        ):
            response = "再见。" if use_chinese else "Goodbye."
        elif self._contains(
            text,
            ("你能看到我吗", "你看得到我吗", "能看到我", "can you see me"),
        ):
            if context.person_detected:
                response = "能，我看到你了。" if use_chinese else "Yes, I can see you."
            else:
                response = (
                    "我现在没有检测到你。"
                    if use_chinese
                    else "I cannot detect you right now."
                )
        elif self._contains(
            text,
            ("你在看我吗", "正在看我", "are you looking at me"),
        ):
            if context.eye_contact == "true":
                response = (
                    "嗯，我正在看着你。"
                    if use_chinese
                    else "Yes, I am looking at you."
                )
            else:
                response = (
                    "我现在没有和你对视。"
                    if use_chinese
                    else "I am not making eye contact with you right now."
                )
        elif (
            self._contains(text, ("你好", "嗨"))
            or text in {"hello", "hi", "hey", "hello npc", "hi npc"}
        ):
            response = "你好。" if use_chinese else "Hello."
        else:
            response = (
                "我听到了，但我现在还不知道该怎么回答。"
                if use_chinese
                else "I heard you, but I do not know how to answer yet."
            )

        return ResponseEvent.create(response, "zh" if use_chinese else "en")

    @staticmethod
    def _contains(text, phrases):
        return any(phrase in text for phrase in phrases)
