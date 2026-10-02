"""由连续 Player State 提炼出的离散 World Event。"""

import time
from dataclasses import asdict, dataclass


EVENT_PRIORITIES = {
    "player_enter": "high",
    "player_leave": "high",
    "hand_wave": "medium",
    "player_approach": "medium",
    "player_far": "medium",
    "eye_contact_started": "low",
    "eye_contact_long": "low",
}

PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class WorldEvent:
    message_type: str
    event_type: str
    timestamp: int
    priority: str
    context: dict

    @classmethod
    def create(cls, event_type, context, timestamp=None):
        if event_type not in EVENT_PRIORITIES:
            raise ValueError(f"不支持的 World Event：{event_type}")
        return cls(
            message_type="world_event",
            event_type=event_type,
            timestamp=(
                int(time.time() * 1000)
                if timestamp is None
                else int(timestamp)
            ),
            priority=EVENT_PRIORITIES[event_type],
            context=dict(context),
        )

    def to_dict(self):
        return asdict(self)
