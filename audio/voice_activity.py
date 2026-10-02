"""基于音量的轻量 Voice Activity Detection。"""

import time

from .audio_config import (
    VAD_HANGOVER,
    VAD_MIN_SPEECH_DURATION,
    VAD_START_THRESHOLD,
    VAD_STOP_THRESHOLD,
)


class VoiceActivityDetector:
    """使用双阈值、最短持续时间和 hangover 抑制状态闪烁。"""

    def __init__(self):
        self.is_speaking = False
        self._above_since = None
        self._last_voice_time = None

    def reset(self):
        self.is_speaking = False
        self._above_since = None
        self._last_voice_time = None

    def update(self, rms, now=None):
        now = time.monotonic() if now is None else now

        if self.is_speaking:
            if rms >= VAD_STOP_THRESHOLD:
                self._last_voice_time = now
            elif (
                self._last_voice_time is not None
                and now - self._last_voice_time >= VAD_HANGOVER
            ):
                self.is_speaking = False
                self._above_since = None
            return self.is_speaking

        if rms >= VAD_START_THRESHOLD:
            if self._above_since is None:
                self._above_since = now
            if now - self._above_since >= VAD_MIN_SPEECH_DURATION:
                self.is_speaking = True
                self._last_voice_time = now
        else:
            self._above_since = None

        return self.is_speaking
