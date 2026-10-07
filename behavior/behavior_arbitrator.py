"""基于规则和分数的确定性 Behavior Arbitration。"""

import time

from emotion.emotion_state import EmotionState

from .behavior_candidate import BehaviorCandidate, BehaviorDecision
from .cooldown import ARBITRATION_COOLDOWNS, CooldownTracker


# 分数相同时，越靠前的行为优先级越高。禁止使用随机数。
TIE_BREAK_ORDER = (
    "listen",
    "speak",
    "look_at_player",
    "greet",
    "wave",
    "observe",
    "idle",
)
_TIE_BREAK_RANK = {
    action: index for index, action in enumerate(TIE_BREAK_ORDER)
}

ATTENTION_BIAS = {
    "normal": 0.00,
    "slightly_high": 0.03,
    "high": 0.06,
}


class BehaviorArbitrator:
    """只选择当前行为；不会调用 LLM、Unity 或其他执行器。"""

    def __init__(self, cooldowns=None, clock=None):
        self.cooldowns = cooldowns or CooldownTracker(ARBITRATION_COOLDOWNS)
        self.clock = clock or time.monotonic

    def select(
        self,
        perception=None,
        relationship_behavior=None,
        emotion_state=None,
        conversation_active=False,
        personality=None,
        behavior_state="idle",
        event_type=None,
        now=None,
    ):
        """生成候选、应用 Hard Gates，并返回最高分合法行为。"""
        now = self.clock() if now is None else float(now)
        emotion = (
            emotion_state
            if isinstance(emotion_state, EmotionState)
            else EmotionState()
        )
        state = str(behavior_state or "idle").strip().lower()
        player_visible = _truthy(
            _read(perception, "player_visible", None)
        ) or _truthy(_read(perception, "person_detected", False))
        eye_contact = _truthy(_read(perception, "eye_contact", False))
        speech_detected = _truthy(
            _read(perception, "speech_detected", False)
        ) or _truthy(_read(perception, "is_speaking", False))
        player_entered = (
            str(event_type or "") == "player_enter"
            or _truthy(_read(perception, "player_entered", False))
        )
        player_waving = (
            str(event_type or "") == "hand_wave"
            or _truthy(_read(perception, "player_waving", False))
            or str(_read(perception, "left_hand", "down")) == "raised"
            or str(_read(perception, "right_hand", "down")) == "raised"
        )

        allow_greeting = bool(
            _read(
                relationship_behavior,
                "allow_proactive_greeting",
                False,
            )
        )
        allow_wave = bool(
            _read(
                relationship_behavior,
                "allow_proactive_wave",
                False,
            )
        )
        attention = str(
            _read(relationship_behavior, "attention_priority", "normal")
        )
        attention_bias = ATTENTION_BIAS.get(attention, 0.0)
        calm_personality = "calm" in _personality_traits(personality)
        busy = state in {"listening", "thinking", "speaking", "greeting"}

        candidates = [
            BehaviorCandidate("idle", 0.10, ("safe_fallback",)),
            _listen_candidate(state, speech_detected),
            _speak_candidate(state),
            _look_candidate(
                player_visible,
                eye_contact,
                emotion,
                attention_bias,
            ),
            _observe_candidate(player_visible, emotion, calm_personality),
            self._greet_candidate(
                player_visible,
                player_entered,
                allow_greeting,
                bool(conversation_active),
                busy,
                emotion,
                attention_bias,
                calm_personality,
                now,
            ),
            self._wave_candidate(
                player_visible,
                player_waving,
                allow_wave,
                bool(conversation_active),
                busy,
                emotion,
                attention_bias,
                calm_personality,
                now,
            ),
        ]
        selected = select_highest(candidates)
        if selected.action in ARBITRATION_COOLDOWNS:
            self.cooldowns.trigger(selected.action, now)
        return BehaviorDecision(
            action=selected.action,
            score=selected.score,
            reasons=selected.reasons,
            candidates=tuple(candidates),
        )

    def _greet_candidate(
        self,
        visible,
        entered,
        allowed,
        conversation_active,
        busy,
        emotion,
        attention_bias,
        calm_personality,
        now,
    ):
        reasons = []
        gates = []
        if not visible:
            gates.append("player_not_visible")
        if not entered:
            gates.append("no_player_enter_event")
        else:
            reasons.append("player_enter_event")
        if not allowed:
            gates.append("relationship_blocks_greeting")
        if conversation_active:
            gates.append("active_conversation")
        if busy:
            gates.append("npc_busy")
        if not self.cooldowns.is_ready("greet", now):
            gates.append("greet_cooldown_active")

        score = 0.38 + 0.18 + attention_bias
        score = _social_score(score, emotion, reasons)
        if calm_personality:
            score -= 0.03
            reasons.append("calm_personality_bias")
        reasons.extend(gates)
        return BehaviorCandidate("greet", score, tuple(reasons), not gates)

    def _wave_candidate(
        self,
        visible,
        player_waving,
        allowed,
        conversation_active,
        busy,
        emotion,
        attention_bias,
        calm_personality,
        now,
    ):
        reasons = []
        gates = []
        if not visible:
            gates.append("player_not_visible")
        if not player_waving:
            gates.append("no_player_wave_event")
        else:
            reasons.append("player_wave_event")
        if not allowed:
            gates.append("relationship_blocks_wave")
        if conversation_active:
            gates.append("active_conversation")
        if busy:
            gates.append("npc_busy")
        if not self.cooldowns.is_ready("wave", now):
            gates.append("wave_cooldown_active")

        score = 0.40 + 0.20 + attention_bias
        score = _social_score(score, emotion, reasons)
        if calm_personality:
            score -= 0.03
            reasons.append("calm_personality_bias")
        reasons.extend(gates)
        return BehaviorCandidate("wave", score, tuple(reasons), not gates)


def select_highest(candidates):
    """公开的确定性选择函数，便于独立测试 tie-breaking。"""
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise ValueError("至少需要一个合法 Behavior Candidate。")
    return max(
        eligible,
        key=lambda candidate: (
            candidate.score,
            -_TIE_BREAK_RANK[candidate.action],
        ),
    )


def _listen_candidate(state, speech_detected):
    eligible = state == "listening" or speech_detected
    reasons = (
        ("speech_detected", "player_input_has_highest_priority")
        if eligible
        else ("no_active_speech",)
    )
    return BehaviorCandidate("listen", 1.0, reasons, eligible)


def _speak_candidate(state):
    eligible = state == "speaking"
    reasons = ("npc_is_speaking",) if eligible else ("npc_not_speaking",)
    return BehaviorCandidate("speak", 0.98, reasons, eligible)


def _look_candidate(visible, eye_contact, emotion, attention_bias):
    reasons = ["player_visible"]
    score = 0.47 + emotion.social_comfort * 0.15 + attention_bias
    if eye_contact:
        score += 0.25
        reasons.append("eye_contact")
    if emotion.social_comfort >= 0.60:
        reasons.append("high_social_comfort")
    if not visible:
        reasons.append("player_not_visible")
    return BehaviorCandidate("look_at_player", score, tuple(reasons), visible)


def _observe_candidate(visible, emotion, calm_personality):
    reasons = ["player_visible", "curiosity_bias"]
    score = 0.25 + emotion.curiosity * 0.40
    if calm_personality:
        score += 0.02
        reasons.append("calm_observant_bias")
    if not visible:
        reasons.append("player_not_visible")
    return BehaviorCandidate("observe", score, tuple(reasons), visible)


def _social_score(score, emotion, reasons):
    if emotion.valence > 0.20:
        reasons.append("positive_emotion")
    score += max(0.0, emotion.valence) * 0.12
    score += emotion.social_comfort * 0.12
    if emotion.social_comfort >= 0.60:
        reasons.append("high_social_comfort")
    if emotion.social_comfort < 0.30:
        score -= 0.18
        reasons.append("low_social_comfort")
    return score


def _read(source, name, default):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "raised"}
    return bool(value)


def _personality_traits(personality):
    values = _read(personality, "traits", ())
    return {str(value).strip().lower() for value in values}
