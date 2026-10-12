"""统一调度现有 NPC 模块；业务规则仍由各模块负责。"""

from dataclasses import dataclass

from behavior.autonomous_controller import AutonomousController
from behavior.behavior_manager import BehaviorManager, VALID_STATES
from behavior.relationship_behavior import build_relationship_behavior_profile
from emotion.emotion_engine import EmotionEngine


@dataclass(frozen=True)
class RuntimeResult:
    events_processed: tuple
    emotion_snapshot: dict
    relationship_update: dict
    autonomous_decision: object
    dispatch_actions: tuple[str, ...]
    error: str | None = None


class NPCManager:
    """显式 tick，无后台线程；MemoryStore 由调用方注入。"""

    def __init__(self, emotion_engine=None, memory_store=None,
                 behavior_manager=None, controller=None):
        self.emotion_engine = emotion_engine or EmotionEngine()
        self.memory_store = memory_store
        self.behavior_manager = behavior_manager or BehaviorManager()
        self.controller = controller or AutonomousController(self.behavior_manager)
        self._processed = []

    def tick(self, player_state=None, world_events=(), emotion_state=None,
             relationship_state=None, conversation_state=False,
             personality=None, behavior_state=None, now=None):
        """处理事件，再把最新只读状态交给已有 Controller。"""
        stage = "inputs"
        processed = []
        relationship_update = {}
        try:
            if player_state is None:
                player_state = {}
            if behavior_state is not None:
                if behavior_state not in VALID_STATES:
                    raise ValueError("invalid behavior state")
                self.behavior_manager.state = behavior_state
            events = tuple(world_events or ())
            stage = "emotion"
            self.emotion_engine.update(now=now)
            for event in events:
                # 同一 WorldEvent 重复交付时不重复更新 Emotion/Memory。
                if any(item is event for item in self._processed):
                    continue
                self.emotion_engine.process_world_event(event, now=now)
                stage = "memory"
                if self.memory_store is not None:
                    self.memory_store.observe_world_events([event])
                self._processed.append(event)
                self._processed = self._processed[-256:]
                processed.append(event.event_type)
                stage = "emotion"
            relationship = relationship_state
            if self.memory_store is not None:
                relationship_update = self.memory_store.profile.to_dict()
                relationship = self.memory_store.profile
            if hasattr(relationship, "allow_proactive_wave"):
                profile = relationship
            else:
                level = _value(relationship, "relationship_level", "stranger")
                if isinstance(relationship, str):
                    level = relationship
                profile = build_relationship_behavior_profile(level)
            stage = "controller"
            decisions = []
            actions = []
            # 逐个交付已有事件；Controller 保持双通道和重复抑制语义。
            for event in events or (None,):
                decision = self.controller.tick(
                    perception=player_state,
                    relationship_behavior=profile,
                    emotion_state=(emotion_state if emotion_state is not None
                                   else self.emotion_engine.state),
                    conversation_active=bool(_value(
                        conversation_state, "active", conversation_state)),
                    personality=personality,
                    latest_world_event=event,
                    now=now,
                )
                if decision.fallback:
                    raise ValueError("controller fallback")
                decisions.append(decision)
                actions.extend(decision.dispatches)
            return RuntimeResult(
                tuple(processed), self.emotion_engine.state.snapshot(),
                relationship_update, decisions[-1],
                tuple(dict.fromkeys(actions)),
            )
        except Exception:
            # 已完成的 Memory 写入不会回滚；失败 tick 不交付任何社交动作。
            return RuntimeResult(tuple(processed), {}, relationship_update,
                                 None, ("idle",), f"{stage}_failed")


def _value(source, name, default):
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)
