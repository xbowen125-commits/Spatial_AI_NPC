"""连续 Emotion State、派生标签与固定步长衰减。"""

from dataclasses import asdict, dataclass

from .emotion_config import (
    BASELINE_AROUSAL,
    BASELINE_CURIOSITY,
    BASELINE_SOCIAL_COMFORT,
    BASELINE_VALENCE,
    EMOTION_DECAY_STEP,
)


@dataclass(frozen=True)
class EmotionState:
    valence: float = BASELINE_VALENCE
    arousal: float = BASELINE_AROUSAL
    social_comfort: float = BASELINE_SOCIAL_COMFORT
    curiosity: float = BASELINE_CURIOSITY

    @classmethod
    def create(
        cls,
        valence=BASELINE_VALENCE,
        arousal=BASELINE_AROUSAL,
        social_comfort=BASELINE_SOCIAL_COMFORT,
        curiosity=BASELINE_CURIOSITY,
    ):
        return cls(
            valence=clamp(float(valence), -1.0, 1.0),
            arousal=clamp(float(arousal), 0.0, 1.0),
            social_comfort=clamp(float(social_comfort), 0.0, 1.0),
            curiosity=clamp(float(curiosity), 0.0, 1.0),
        )

    def apply_delta(self, delta):
        return EmotionState.create(
            self.valence + delta.valence,
            self.arousal + delta.arousal,
            self.social_comfort + delta.social_comfort,
            self.curiosity + delta.curiosity,
        )

    def decay_toward_baseline(self, step=EMOTION_DECAY_STEP):
        step = max(0.0, float(step))
        baseline = EmotionState()
        return EmotionState.create(
            _move_toward(self.valence, baseline.valence, step),
            _move_toward(self.arousal, baseline.arousal, step),
            _move_toward(
                self.social_comfort,
                baseline.social_comfort,
                step,
            ),
            _move_toward(self.curiosity, baseline.curiosity, step),
        )

    def labels(self):
        """标签仅用于解释连续状态，不作为状态存储。"""
        labels = []
        if self.arousal <= 0.35 and self.valence >= 0.0:
            labels.append("calm")
        if self.valence >= 0.30 and self.social_comfort >= 0.48:
            labels.append("warm")
        if self.curiosity >= 0.52:
            labels.append("curious")
        if self.valence <= -0.10 or self.social_comfort <= 0.25:
            labels.append("uneasy")
        if self.arousal >= 0.70 and self.valence >= 0.20:
            labels.append("excited")
        return tuple(labels or ("balanced",))

    def snapshot(self):
        data = asdict(self)
        data["labels"] = list(self.labels())
        return data

    def to_context(self):
        return "\n".join(
            (
                "EMOTION CONTEXT:",
                f"valence={self.valence:.2f}",
                f"arousal={self.arousal:.2f}",
                f"social_comfort={self.social_comfort:.2f}",
                f"curiosity={self.curiosity:.2f}",
                f"labels={','.join(self.labels())}",
                "Emotion may lightly influence expression, but must not "
                "override safety, personality, relationship, or perception.",
            )
        )

    @property
    def emotion(self):
        """为现有 Unity/Behavior 接口提供派生兼容标签。"""
        labels = self.labels()
        if "uneasy" in labels:
            return "surprised"
        if "excited" in labels or "warm" in labels:
            return "happy"
        if "curious" in labels:
            return "curious"
        return "neutral"

    @property
    def intensity(self):
        """以相对 baseline 的最大归一化偏移作为兼容强度。"""
        distances = (
            abs(self.valence - BASELINE_VALENCE) / 1.2,
            abs(self.arousal - BASELINE_AROUSAL) / 0.7,
            abs(self.social_comfort - BASELINE_SOCIAL_COMFORT) / 0.6,
            abs(self.curiosity - BASELINE_CURIOSITY) / 0.5,
        )
        return clamp(max(distances), 0.0, 1.0)

    def to_dict(self):
        data = self.snapshot()
        data["emotion"] = self.emotion
        data["intensity"] = self.intensity
        return data


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _move_toward(value, target, step):
    if value < target:
        return min(target, value + step)
    if value > target:
        return max(target, value - step)
    return target
