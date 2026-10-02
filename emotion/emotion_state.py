"""Emotion State Schema 与数值Clamp。"""

import time
from dataclasses import asdict, dataclass

from .emotion_config import ALLOWED_EMOTIONS


@dataclass(frozen=True)
class EmotionState:
    emotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    intensity: float = 0.0
    cause: str = "none"
    updated_at: int = 0

    @classmethod
    def create(
        cls,
        emotion="neutral",
        valence=0.0,
        arousal=0.0,
        intensity=0.0,
        cause="none",
        updated_at=None,
    ):
        emotion = str(emotion).lower()
        if emotion not in ALLOWED_EMOTIONS:
            emotion = "neutral"
        return cls(
            emotion=emotion,
            valence=clamp(float(valence), -1.0, 1.0),
            arousal=clamp(float(arousal), 0.0, 1.0),
            intensity=clamp(float(intensity), 0.0, 1.0),
            cause=str(cause),
            updated_at=(
                int(time.time() * 1000)
                if updated_at is None
                else int(updated_at)
            ),
        )

    def to_dict(self):
        return asdict(self)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))
