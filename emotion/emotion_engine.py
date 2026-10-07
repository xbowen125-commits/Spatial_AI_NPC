"""不依赖 LLM 的确定性 Dynamic Emotion Engine。"""

import time

from .emotion_config import EMOTION_DECAY_STEP, EMOTION_UPDATE_INTERVAL
from .emotion_policy import (
    delta_for_event,
    emotion_event_for_world_event,
)
from .emotion_state import EmotionState


class EmotionEngine:
    def __init__(self, logger=None, now=None):
        self.logger = logger
        self.state = EmotionState()
        self._last_update = time.monotonic() if now is None else float(now)

    def apply_event(
        self,
        event_type,
        relationship_level="unchanged",
        trigger_event=None,
        now=None,
    ):
        """按固定 delta 更新；relationship 只写日志，不改变数值。"""
        delta = delta_for_event(event_type)
        if delta is None:
            return self.state
        previous = self.state
        self.state = self.state.apply_delta(delta)
        self._last_update = (
            time.monotonic() if now is None else float(now)
        )
        self._log(trigger_event or event_type, previous, relationship_level)
        return self.state

    def decay_toward_baseline(self, step=EMOTION_DECAY_STEP, now=None):
        previous = self.state
        self.state = self.state.decay_toward_baseline(step)
        self._last_update = (
            time.monotonic() if now is None else float(now)
        )
        if self.state != previous:
            self._log("decay_toward_baseline", previous, "unchanged")
        return self.state

    def process_world_event(
        self,
        world_event,
        relationship_level="stranger",
        now=None,
    ):
        event_type = emotion_event_for_world_event(world_event.event_type)
        if event_type is None:
            return self.state
        return self.apply_event(
            event_type,
            relationship_level=relationship_level,
            trigger_event=world_event.event_type,
            now=now,
        )

    def process_speech_event(self, relationship_level="stranger", now=None):
        """不读取 Speech Text；一次有效对话视为 friendly_conversation。"""
        return self.apply_event(
            "friendly_conversation",
            relationship_level=relationship_level,
            trigger_event="speech_event",
            now=now,
        )

    def update(self, now=None, force=False):
        """兼容主循环：到固定间隔时执行一次固定步长 decay。"""
        now = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, now - self._last_update)
        if not force and elapsed < EMOTION_UPDATE_INTERVAL:
            return self.state
        return self.decay_toward_baseline(now=now)

    def _log(self, trigger_event, previous, relationship_level):
        if self.logger is not None:
            self.logger.write(
                trigger_event=trigger_event,
                previous_state=previous,
                new_state=self.state,
                relationship_level=relationship_level,
            )
