"""主动行为状态机、优先级、冲突处理与规则调度。"""

import time

from events.world_event import PRIORITY_RANK

from .behavior_arbitrator import BehaviorArbitrator
from .behavior_rules import BehaviorRules
from .cooldown import BEHAVIOR_STATE_COOLDOWN, CooldownTracker
from .relationship_behavior import build_relationship_behavior_profile


BUSY_STATES = {"listening", "thinking", "speaking", "greeting", "cooldown"}
VALID_STATES = {
    "idle",
    "observing",
    "greeting",
    "listening",
    "thinking",
    "speaking",
    "cooldown",
}


class BehaviorManager:
    def __init__(
        self,
        rules=None,
        cooldowns=None,
        logger=None,
        arbitrator=None,
    ):
        self.rules = rules or BehaviorRules()
        self.cooldowns = cooldowns or CooldownTracker()
        self.arbitrator = arbitrator or BehaviorArbitrator()
        self.logger = logger
        self.state = "idle"
        self._pending_event = None
        self._cooldown_until = 0.0
        self._player_present = False

    def arbitrate(
        self,
        perception=None,
        relationship_behavior=None,
        emotion_state=None,
        conversation_active=False,
        personality=None,
        event_type=None,
        now=None,
    ):
        """把只读运行时上下文交给仲裁层；不执行 Unity 动作。"""
        return self.arbitrator.select(
            perception=perception,
            relationship_behavior=relationship_behavior,
            emotion_state=emotion_state,
            conversation_active=conversation_active,
            personality=personality,
            behavior_state=self.state,
            event_type=event_type,
            now=now,
        )

    def observe_player(self, person_detected):
        self._player_present = bool(person_detected)
        if self.state == "idle" and self._player_present:
            self._set_state("observing")
        elif self.state == "observing" and not self._player_present:
            self._set_state("idle")

    def handle_events(self, events, now=None):
        now = time.monotonic() if now is None else float(now)
        if not events:
            return None

        ordered = sorted(
            events,
            key=lambda event: PRIORITY_RANK[event.priority],
            reverse=True,
        )
        selected = ordered[0]
        for ignored in ordered[1:]:
            self._log(ignored, None, "superseded")

        if self.state in BUSY_STATES:
            queue_status = self._queue_highest(selected)
            self._log(selected, None, queue_status)
            return None
        return self._execute(selected, now)

    def tick(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if self.state == "cooldown" and now >= self._cooldown_until:
            self._set_state("observing" if self._player_present else "idle")
        if self.state not in {"idle", "observing"} or self._pending_event is None:
            return None
        event = self._pending_event
        self._pending_event = None
        return self._execute(event, now)

    def on_speech_event(self):
        """玩家主动输入优先级最高：取消尚未执行的主动行为。"""
        if self._pending_event is not None:
            self._log(self._pending_event, None, "cancelled_by_speech")
        self._pending_event = None
        self._set_state("listening")

    def discard_events(self, events, reason="cancelled_by_speech"):
        for event in events:
            self._log(event, None, reason)

    def on_agent_started(self):
        self._set_state("thinking")

    def on_response_generated(self):
        self._set_state("speaking")

    def on_tts_started(self):
        if self.state != "greeting":
            self._set_state("speaking")

    def on_tts_finished(self, now=None):
        now = time.monotonic() if now is None else float(now)
        self._set_state("cooldown")
        self._cooldown_until = now + BEHAVIOR_STATE_COOLDOWN

    def on_text_only_finished(self, now=None):
        self.on_tts_finished(now)

    def _execute(self, event, now):
        if not self._relationship_allows(event):
            self._log(event, None, "relationship_blocked")
            return None

        if not self.cooldowns.is_ready(event.event_type, now):
            self._log(
                event,
                None,
                "cooldown_blocked",
                self.cooldowns.remaining(event.event_type, now),
            )
            return None

        decision = self.rules.decide(event)
        if decision is None:
            self._log(event, None, "no_rule")
            return None

        self.cooldowns.trigger(event.event_type, now)
        # 进入问候时同时压制同一动作里的 hand_wave，避免欢迎和挥手重复。
        if event.event_type == "player_enter":
            self.cooldowns.trigger("hand_wave", now)
            self._set_state("greeting")
        elif decision.reply:
            self._set_state("speaking")
        else:
            self._set_state("cooldown")
            self._cooldown_until = now + BEHAVIOR_STATE_COOLDOWN
        self._log(event, decision, "executed")
        return decision

    @staticmethod
    def _relationship_allows(event):
        """只在 Memory 已明确提供关系等级时限制主动社交行为。"""
        if "relationship_level" not in event.context:
            return True
        profile = build_relationship_behavior_profile(
            event.context.get("relationship_level"),
            interaction_context=event.context,
        )
        if event.event_type == "player_enter":
            return profile.allow_proactive_greeting
        if event.event_type == "hand_wave":
            return profile.allow_proactive_wave
        return True

    def _queue_highest(self, event):
        if self._pending_event is None:
            self._pending_event = event
            return "queued_busy"
        if PRIORITY_RANK[event.priority] > PRIORITY_RANK[self._pending_event.priority]:
            self._log(self._pending_event, None, "replaced_in_queue")
            self._pending_event = event
            return "queued_replacing_lower_priority"
        return "lower_priority_ignored"

    def _set_state(self, state):
        if state not in VALID_STATES:
            raise ValueError(f"非法 npc_behavior_state：{state}")
        self.state = state

    def _log(self, event, decision, cooldown_status, remaining=0.0):
        if self.logger is not None:
            self.logger.write(
                event,
                decision,
                cooldown_status,
                cooldown_remaining=remaining,
                behavior_state=self.state,
            )
