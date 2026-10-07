"""Behavior Arbitration 使用的不可变候选与决策结构。"""

from dataclasses import asdict, dataclass


SUPPORTED_ACTIONS = (
    "listen",
    "speak",
    "look_at_player",
    "greet",
    "wave",
    "observe",
    "idle",
)


@dataclass(frozen=True)
class BehaviorCandidate:
    """一个带分数、解释和 Hard Gate 结果的候选行为。"""

    action: str
    score: float
    reasons: tuple[str, ...]
    eligible: bool = True

    def __post_init__(self):
        action = str(self.action).strip().lower()
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"不支持的 Arbitration Action：{action}")
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "score",
            round(max(0.0, min(1.0, float(self.score))), 4),
        )
        object.__setattr__(self, "reasons", tuple(map(str, self.reasons)))
        object.__setattr__(self, "eligible", bool(self.eligible))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BehaviorDecision:
    """仲裁结果；不执行动作、不生成语言，也不保存历史。"""

    action: str
    score: float
    reasons: tuple[str, ...]
    candidates: tuple[BehaviorCandidate, ...]

    def to_dict(self):
        return {
            "action": self.action,
            "score": self.score,
            "reasons": list(self.reasons),
            "candidates": [item.to_dict() for item in self.candidates],
        }
