"""明确事件到固定 Emotion Delta 的规则。"""

from dataclasses import dataclass
from dataclasses import replace

from behavior.behavior_rules import BehaviorRules


@dataclass(frozen=True)
class EmotionDelta:
    valence: float = 0.0
    arousal: float = 0.0
    social_comfort: float = 0.0
    curiosity: float = 0.0


EVENT_DELTAS = {
    "friendly_conversation": EmotionDelta(0.10, 0.02, 0.08, 0.02),
    "negative_interaction": EmotionDelta(-0.15, 0.10, -0.10, -0.02),
    "eye_contact": EmotionDelta(0.00, 0.02, 0.03, 0.02),
    "player_wave": EmotionDelta(0.05, 0.05, 0.04, 0.01),
    "player_absent": EmotionDelta(-0.02, -0.03, -0.02, 0.00),
    "long_idle": EmotionDelta(0.00, -0.05, 0.00, -0.02),
}

WORLD_EVENT_TO_EMOTION_EVENT = {
    "player_enter": "friendly_conversation",
    "player_leave": "player_absent",
    "eye_contact_started": "eye_contact",
    "eye_contact_long": "eye_contact",
    "hand_wave": "player_wave",
    "player_far": "player_absent",
}


def delta_for_event(event_type):
    return EVENT_DELTAS.get(str(event_type))


def emotion_event_for_world_event(event_type):
    return WORLD_EVENT_TO_EMOTION_EVENT.get(str(event_type))


class EmotionAwareBehaviorRules:
    """只读取派生标签；不会反向更新 Emotion State。"""

    def __init__(self, state_provider, base_rules=None):
        self.state_provider = state_provider
        self.base_rules = base_rules or BehaviorRules()

    def decide(self, world_event):
        decision = self.base_rules.decide(world_event)
        if decision is None:
            return None
        state = self.state_provider()
        if (
            "curious" in state.labels()
            and world_event.event_type == "eye_contact_long"
            and decision.action == "none"
        ):
            return replace(decision, action="look_at_player")
        return decision
