"""外部事件到 Emotion Stimulus 的规则与轻量Behavior调制。"""

from dataclasses import dataclass, replace

from behavior.behavior_rules import BehaviorRules


@dataclass(frozen=True)
class EmotionStimulus:
    emotion: str
    valence: float
    arousal: float
    intensity: float
    cause: str


WORLD_EVENT_EMOTIONS = {
    "player_enter": EmotionStimulus(
        "happy", 0.45, 0.35, 0.45, "player_enter"
    ),
    "hand_wave": EmotionStimulus(
        "happy", 0.60, 0.50, 0.60, "hand_wave"
    ),
    "eye_contact_started": EmotionStimulus(
        "curious", 0.20, 0.35, 0.35, "eye_contact_started"
    ),
    "eye_contact_long": EmotionStimulus(
        "curious", 0.25, 0.45, 0.50, "eye_contact_long"
    ),
    "player_approach": EmotionStimulus(
        "curious", 0.15, 0.30, 0.30, "player_approach"
    ),
}


def stimulus_for_world_event(event_type):
    return WORLD_EVENT_EMOTIONS.get(str(event_type))


class EmotionAwareBehaviorRules:
    """只读取Emotion State；不会反向更新Emotion或创建World Event。"""

    def __init__(self, state_provider, base_rules=None):
        self.state_provider = state_provider
        self.base_rules = base_rules or BehaviorRules()

    def decide(self, world_event):
        decision = self.base_rules.decide(world_event)
        if decision is None:
            return None
        state = self.state_provider()
        if (
            state.emotion == "curious"
            and world_event.event_type == "eye_contact_long"
            and decision.action == "none"
        ):
            return replace(decision, action="look_at_player")
        return decision
