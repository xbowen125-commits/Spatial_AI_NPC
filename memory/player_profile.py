"""可持久化的结构化玩家关系档案。"""

from dataclasses import asdict, dataclass, field


RELATIONSHIP_LEVELS = ("stranger", "acquaintance", "friend")
MEMORY_ALLOWLIST_FIELDS = (
    "player_id",
    "first_seen",
    "interaction_count",
    "times_seen",
    "times_spoken",
    "times_waved",
    "relationship_level",
    "last_interaction",
    "preferences",
)


@dataclass
class PlayerProfile:
    player_id: str = "local_player"
    first_seen: int = 0
    interaction_count: int = 0
    times_seen: int = 0
    times_spoken: int = 0
    times_waved: int = 0
    relationship_level: str = "stranger"
    last_interaction: int = 0
    preferences: dict = field(default_factory=dict)

    def to_dict(self):
        """只导出明确允许的结构化字段。"""
        data = asdict(self)
        return {field: data[field] for field in MEMORY_ALLOWLIST_FIELDS}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Player Profile 必须是JSON对象。")

        relationship = str(data.get("relationship_level", "stranger"))
        if relationship not in RELATIONSHIP_LEVELS:
            raise ValueError("relationship_level 非法。")

        preferences = data.get("preferences", {})
        if not isinstance(preferences, dict):
            raise ValueError("preferences 必须是JSON对象。")

        return cls(
            player_id=str(data.get("player_id", "local_player")),
            first_seen=_non_negative_int(data.get("first_seen", 0)),
            interaction_count=_non_negative_int(
                data.get("interaction_count", 0)
            ),
            times_seen=_non_negative_int(data.get("times_seen", 0)),
            times_spoken=_non_negative_int(data.get("times_spoken", 0)),
            times_waved=_non_negative_int(data.get("times_waved", 0)),
            relationship_level=relationship,
            last_interaction=_non_negative_int(
                data.get("last_interaction", 0)
            ),
            preferences=dict(preferences),
        )


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise ValueError("计数和时间戳不能为负数。")
    return value
