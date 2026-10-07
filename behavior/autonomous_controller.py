"""决定何时运行仲裁，以及哪些仲裁结果值得执行。"""

from collections import deque
from dataclasses import dataclass

from .behavior_arbitrator import select_highest
from .behavior_candidate import BehaviorCandidate, BehaviorDecision


ATTENTION_ACTIONS = frozenset({"look_at_player", "observe", "idle"})
STATEFUL_INTERACTION_ACTIONS = frozenset({"listen", "speak"})
DISCRETE_EVENT_ACTIONS = {
    "player_enter": "greet",
    "hand_wave": "wave",
}
MAX_CONSUMED_EVENTS = 32


@dataclass(frozen=True)
class TickResult:
    """一次 Tick 的双通道决策和待交给现有执行路径的动作。"""

    attention: str
    interaction: str | None
    executed: bool
    dispatches: tuple[str, ...]
    reasons: tuple[str, ...]
    arbitration: BehaviorDecision
    fallback: bool = False

    def to_dict(self):
        return {
            "attention": self.attention,
            "interaction": self.interaction,
            "executed": self.executed,
            "dispatches": list(self.dispatches),
            "reasons": list(self.reasons),
            "arbitration": self.arbitration.to_dict(),
            "fallback": self.fallback,
        }


class AutonomousController:
    """无后台线程的确定性自主行为协调器。"""

    def __init__(self, behavior_manager):
        self.behavior_manager = behavior_manager
        self._attention = "idle"
        self._stateful_interaction = None
        self._consumed_order = deque()
        self._consumed_tokens = set()

    def tick(
        self,
        perception=None,
        relationship_behavior=None,
        emotion_state=None,
        conversation_active=False,
        personality=None,
        latest_world_event=None,
        now=None,
    ):
        """运行一次仲裁，抑制重复，并返回需要执行的动作。"""
        event_type = _event_value(latest_world_event, "event_type")
        event_token = _event_token(latest_world_event)
        try:
            arbitration = self.behavior_manager.arbitrate(
                perception=perception,
                relationship_behavior=relationship_behavior,
                emotion_state=emotion_state,
                conversation_active=conversation_active,
                personality=personality,
                event_type=event_type,
                now=now,
            )
            return self._coordinate(
                arbitration,
                event_type,
                event_token,
                now,
            )
        except Exception:
            # 单一模块异常不得终止主循环；错误细节交给上层按变化记录。
            return self._safe_fallback()

    def _coordinate(self, arbitration, event_type, event_token, now):
        attention_candidate = select_highest(
            tuple(
                item
                for item in arbitration.candidates
                if item.action in ATTENTION_ACTIONS
            )
        )
        dispatches = []
        reasons = []

        attention_changed = attention_candidate.action != self._attention
        self._attention = attention_candidate.action
        if attention_changed:
            dispatches.append(self._attention)
            reasons.extend(attention_candidate.reasons)
            reasons.append("attention_changed")

        interaction = self._select_interaction(
            arbitration,
            event_type,
            event_token,
        )
        if interaction is not None:
            if interaction.action in STATEFUL_INTERACTION_ACTIONS:
                if interaction.action != self._stateful_interaction:
                    dispatches.append(interaction.action)
                    reasons.extend(interaction.reasons)
                    reasons.append("interaction_state_changed")
                self._stateful_interaction = interaction.action
            else:
                dispatches.append(interaction.action)
                reasons.extend(interaction.reasons)
                reasons.append("new_discrete_event")
                self.behavior_manager.record_arbitration_execution(
                    interaction.action,
                    now,
                )
        elif self._stateful_interaction is not None:
            self._stateful_interaction = None

        if event_token is not None and event_type in DISCRETE_EVENT_ACTIONS:
            self._remember_event(event_token)

        return TickResult(
            attention=self._attention,
            interaction=(None if interaction is None else interaction.action),
            executed=bool(dispatches),
            dispatches=tuple(dispatches),
            reasons=tuple(dict.fromkeys(reasons or ("no_behavior_change",))),
            arbitration=arbitration,
        )

    def _select_interaction(self, arbitration, event_type, event_token):
        stateful = tuple(
            item
            for item in arbitration.candidates
            if item.action in STATEFUL_INTERACTION_ACTIONS and item.eligible
        )
        if stateful:
            return select_highest(stateful)

        expected_action = DISCRETE_EVENT_ACTIONS.get(event_type)
        if expected_action is None or event_token in self._consumed_tokens:
            return None
        matching = tuple(
            item
            for item in arbitration.candidates
            if item.action == expected_action and item.eligible
        )
        return select_highest(matching) if matching else None

    def _remember_event(self, token):
        if token in self._consumed_tokens:
            return
        if len(self._consumed_order) >= MAX_CONSUMED_EVENTS:
            expired = self._consumed_order.popleft()
            self._consumed_tokens.discard(expired)
        self._consumed_order.append(token)
        self._consumed_tokens.add(token)

    def _safe_fallback(self):
        fallback_candidate = BehaviorCandidate(
            "idle",
            0.10,
            ("safe_fallback", "arbitration_error"),
        )
        arbitration = BehaviorDecision(
            action="idle",
            score=fallback_candidate.score,
            reasons=fallback_candidate.reasons,
            candidates=(fallback_candidate,),
        )
        changed = self._attention != "idle"
        self._attention = "idle"
        self._stateful_interaction = None
        return TickResult(
            attention="idle",
            interaction=None,
            executed=changed,
            dispatches=(("idle",) if changed else ()),
            reasons=("safe_fallback", "arbitration_error"),
            arbitration=arbitration,
            fallback=True,
        )


def _event_value(event, name):
    if event is None:
        return None
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _event_token(event):
    if event is None:
        return None
    event_type = _event_value(event, "event_type")
    timestamp = _event_value(event, "timestamp")
    if event_type is None:
        return None
    # 正常 WorldEvent 总有 timestamp；缺失时同类型事件安全视为同一事件。
    return str(event_type), timestamp
