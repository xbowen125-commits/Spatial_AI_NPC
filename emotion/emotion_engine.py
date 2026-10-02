"""不依赖LLM的本地Emotion Engine。"""

import math
import time

from .emotion_config import (
    EMOTION_DECAY_SECONDS,
    EMOTION_NEUTRAL_THRESHOLD,
    EMOTION_UPDATE_INTERVAL,
    NEW_EMOTION_WEIGHT,
    OLD_EMOTION_WEIGHT,
    RELATIONSHIP_MULTIPLIERS,
    SPEECH_AROUSAL_BOOST,
)
from .emotion_policy import stimulus_for_world_event
from .emotion_state import EmotionState, clamp


class EmotionEngine:
    def __init__(self, logger=None, now=None):
        now = time.monotonic() if now is None else float(now)
        self.logger = logger
        self.state = EmotionState.create()
        self._last_update = now

    def process_world_event(self, world_event, relationship_level="stranger", now=None):
        now = time.monotonic() if now is None else float(now)
        self.update(now, force=True)
        stimulus = stimulus_for_world_event(world_event.event_type)
        if stimulus is None:
            return self.state
        return self.apply_stimulus(
            stimulus,
            relationship_level=relationship_level,
            trigger_event=world_event.event_type,
            now=now,
        )

    def process_speech_event(self, relationship_level="stranger", now=None):
        """Speech只轻微提升arousal，不读取或分析Speech Text。"""
        now = time.monotonic() if now is None else float(now)
        self.update(now, force=True)
        previous = self.state
        self.state = EmotionState.create(
            emotion=previous.emotion,
            valence=previous.valence,
            arousal=previous.arousal + SPEECH_AROUSAL_BOOST,
            intensity=previous.intensity,
            cause=previous.cause,
            updated_at=_timestamp_ms(now),
        )
        self._last_update = now
        self._log("speech_event", previous, relationship_level)
        return self.state

    def apply_stimulus(
        self,
        stimulus,
        relationship_level="stranger",
        trigger_event=None,
        now=None,
    ):
        now = time.monotonic() if now is None else float(now)
        previous = self.state
        multiplier = RELATIONSHIP_MULTIPLIERS.get(
            str(relationship_level),
            RELATIONSHIP_MULTIPLIERS["stranger"],
        )
        target_intensity = clamp(stimulus.intensity * multiplier, 0.0, 1.0)

        if previous.emotion == "neutral" or previous.intensity < EMOTION_NEUTRAL_THRESHOLD:
            valence = stimulus.valence
            arousal = stimulus.arousal
            intensity = target_intensity
        else:
            valence = (
                previous.valence * OLD_EMOTION_WEIGHT
                + stimulus.valence * NEW_EMOTION_WEIGHT
            )
            arousal = (
                previous.arousal * OLD_EMOTION_WEIGHT
                + stimulus.arousal * NEW_EMOTION_WEIGHT
            )
            intensity = (
                previous.intensity * OLD_EMOTION_WEIGHT
                + target_intensity * NEW_EMOTION_WEIGHT
            )

        self.state = EmotionState.create(
            emotion=stimulus.emotion,
            valence=valence,
            arousal=arousal,
            intensity=intensity,
            cause=stimulus.cause,
            updated_at=_timestamp_ms(now),
        )
        self._last_update = now
        self._log(trigger_event or stimulus.cause, previous, relationship_level)
        return self.state

    def update(self, now=None, force=False):
        now = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, now - self._last_update)
        if not force and elapsed < EMOTION_UPDATE_INTERVAL:
            return self.state
        if elapsed <= 0.0 or self.state.emotion == "neutral":
            self._last_update = now
            return self.state

        previous = self.state
        # 30秒后保留约5%，从而稳定越过neutral阈值。
        factor = math.pow(0.05, elapsed / EMOTION_DECAY_SECONDS)
        intensity = previous.intensity * factor
        if intensity < EMOTION_NEUTRAL_THRESHOLD:
            self.state = EmotionState.create(
                updated_at=_timestamp_ms(now),
            )
            trigger = "decay_to_neutral"
        else:
            self.state = EmotionState.create(
                emotion=previous.emotion,
                valence=previous.valence * factor,
                arousal=previous.arousal * factor,
                intensity=intensity,
                cause=previous.cause,
                updated_at=_timestamp_ms(now),
            )
            trigger = None
        self._last_update = now
        if trigger:
            self._log(trigger, previous, "unchanged")
        return self.state

    def _log(self, trigger_event, previous, relationship_level):
        if self.logger is not None:
            self.logger.write(
                trigger_event=trigger_event,
                previous_state=previous,
                new_state=self.state,
                relationship_level=relationship_level,
            )


def _timestamp_ms(now):
    # updated_at 使用Unix时间；now仅用于可测试的单调衰减计时。
    return int(time.time() * 1000)
