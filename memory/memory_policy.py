"""Memory Candidate 保存策略与关系升级规则。"""

import time
from dataclasses import dataclass, field

from behavior.behavior_rules import BehaviorDecision, BehaviorRules


# 所有升级参数集中在这里，方便后续根据 Evaluation 数据调整。
ACQUAINTANCE_MIN_TIMES_SEEN = 2
ACQUAINTANCE_MIN_TIMES_SPOKEN = 3
ACQUAINTANCE_MIN_INTERACTIONS = 3

FRIEND_MIN_TIMES_SEEN = 5
FRIEND_MIN_TIMES_SPOKEN = 5
FRIEND_MIN_INTERACTIONS = 10

AUTO_SAVE_EVENTS = {
    "first_meeting",
    "interaction_count",
    "times_seen",
    "player_waved",
}
CONFIRMATION_REQUIRED_EVENTS = {"player_preference"}


@dataclass(frozen=True)
class MemoryCandidate:
    type: str
    event: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPolicyDecision:
    should_save: bool
    reason: str


class MemoryPolicy:
    def evaluate(self, candidate, confirmed=False):
        if not isinstance(candidate, MemoryCandidate):
            return MemoryPolicyDecision(False, "invalid_candidate")
        if candidate.event in AUTO_SAVE_EVENTS:
            return MemoryPolicyDecision(True, "auto_save")
        if candidate.event in CONFIRMATION_REQUIRED_EVENTS:
            return MemoryPolicyDecision(
                bool(confirmed),
                "confirmed" if confirmed else "confirmation_required",
            )
        return MemoryPolicyDecision(False, "not_allowed")


def calculate_relationship_level(profile):
    """关系只能随结构化计数升级，不读取聊天内容。"""
    current_rank = {"stranger": 0, "acquaintance": 1, "friend": 2}.get(
        profile.relationship_level,
        0,
    )
    if (
        profile.times_seen >= FRIEND_MIN_TIMES_SEEN
        and profile.times_spoken >= FRIEND_MIN_TIMES_SPOKEN
        and profile.interaction_count >= FRIEND_MIN_INTERACTIONS
    ):
        calculated = "friend"
    elif (
        profile.times_seen >= ACQUAINTANCE_MIN_TIMES_SEEN
        or profile.times_spoken >= ACQUAINTANCE_MIN_TIMES_SPOKEN
        or profile.interaction_count >= ACQUAINTANCE_MIN_INTERACTIONS
    ):
        calculated = "acquaintance"
    else:
        calculated = "stranger"
    calculated_rank = {"stranger": 0, "acquaintance": 1, "friend": 2}[
        calculated
    ]
    return profile.relationship_level if current_rank > calculated_rank else calculated


class RelationshipBehaviorRules:
    """在不修改 BehaviorManager 的情况下加入首次见面问候。"""

    def __init__(self, base_rules=None):
        self.base_rules = base_rules or BehaviorRules()

    def decide(self, world_event):
        if (
            world_event.event_type == "player_enter"
            and world_event.context.get("is_first_meeting") is True
        ):
            return BehaviorDecision.create(
                reply="你好，第一次见面。",
                action="wave",
                emotion="happy",
            )
        return self.base_rules.decide(world_event)
