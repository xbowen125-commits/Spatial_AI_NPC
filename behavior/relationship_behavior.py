"""把长期关系等级转换为只读的 NPC 行为策略。"""

from dataclasses import asdict, dataclass


VALID_RELATIONSHIP_LEVELS = {"stranger", "acquaintance", "friend"}


@dataclass(frozen=True)
class RelationshipBehaviorProfile:
    """纯策略结果；不执行动作，也不修改 Memory。"""

    relationship_level: str
    allow_proactive_greeting: bool
    allow_proactive_wave: bool
    preferred_social_distance: str
    response_tone: str
    attention_priority: str

    def to_dict(self):
        return asdict(self)


_PROFILES = {
    "stranger": RelationshipBehaviorProfile(
        relationship_level="stranger",
        allow_proactive_greeting=False,
        allow_proactive_wave=False,
        preferred_social_distance="normal",
        response_tone="polite",
        attention_priority="normal",
    ),
    "acquaintance": RelationshipBehaviorProfile(
        relationship_level="acquaintance",
        allow_proactive_greeting=True,
        allow_proactive_wave=True,
        preferred_social_distance="familiar",
        response_tone="friendly",
        attention_priority="slightly_high",
    ),
    "friend": RelationshipBehaviorProfile(
        relationship_level="friend",
        allow_proactive_greeting=True,
        allow_proactive_wave=True,
        preferred_social_distance="close",
        response_tone="warm",
        attention_priority="high",
    ),
}


def build_relationship_behavior_profile(
    relationship_level,
    perception_state=None,
    interaction_context=None,
):
    """返回关系策略；非法或缺失等级安全降级为 stranger。

    perception_state 和 interaction_context 是为现有调用方预留的只读输入，
    v1.1 默认策略不读取也不保存它们，避免产生新的长期记忆。
    """
    del perception_state, interaction_context
    normalized = str(relationship_level or "stranger").strip().lower()
    if normalized not in VALID_RELATIONSHIP_LEVELS:
        normalized = "stranger"
    return _PROFILES[normalized]
