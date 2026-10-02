"""World Event 到 Behavior Decision 的第一版纯规则映射。"""

import time
from dataclasses import asdict, dataclass


ALLOWED_BEHAVIOR_ACTIONS = {"none", "wave", "nod", "look_at_player"}
ALLOWED_BEHAVIOR_EMOTIONS = {"neutral", "happy", "curious", "surprised"}


@dataclass(frozen=True)
class BehaviorDecision:
    message_type: str
    event_type: str
    timestamp: int
    reply: str
    action: str
    emotion: str
    source: str

    @classmethod
    def create(cls, reply="", action="none", emotion="neutral"):
        action = str(action).strip().lower()
        emotion = str(emotion).strip().lower()
        if action not in ALLOWED_BEHAVIOR_ACTIONS:
            action = "none"
        if emotion not in ALLOWED_BEHAVIOR_EMOTIONS:
            emotion = "neutral"
        return cls(
            message_type="interaction_event",
            event_type="npc_behavior",
            timestamp=int(time.time() * 1000),
            reply=str(reply),
            action=action,
            emotion=emotion,
            source="behavior_rule",
        )

    def to_dict(self):
        return asdict(self)


class BehaviorRules:
    """接口未来可替换成 Planner；当前实现绝不调用 LLM。"""

    def decide(self, world_event):
        rules = {
            "player_enter": BehaviorDecision.create(
                reply="欢迎回来。",
                action="wave",
                emotion="happy",
            ),
            "player_leave": BehaviorDecision.create(reply="再见。"),
            "eye_contact_long": BehaviorDecision.create(
                reply="有什么想问我的吗？",
                emotion="curious",
            ),
            "hand_wave": BehaviorDecision.create(
                reply="你好。",
                action="wave",
                emotion="happy",
            ),
            "player_approach": BehaviorDecision.create(
                action="look_at_player",
                emotion="curious",
            ),
        }
        return rules.get(world_event.event_type)
