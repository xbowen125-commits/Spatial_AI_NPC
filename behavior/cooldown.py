"""主动行为冷却配置与通用计时器。"""


PLAYER_ENTER_COOLDOWN = 60.0
EYE_CONTACT_COOLDOWN = 30.0
HAND_WAVE_COOLDOWN = 5.0
GREET_ACTION_COOLDOWN = 60.0
WAVE_ACTION_COOLDOWN = 5.0

# 状态机中的 cooldown 只是一次行为结束后的短暂过渡，不等同于事件冷却。
BEHAVIOR_STATE_COOLDOWN = 0.5

EVENT_COOLDOWNS = {
    "player_enter": PLAYER_ENTER_COOLDOWN,
    "eye_contact_started": EYE_CONTACT_COOLDOWN,
    "eye_contact_long": EYE_CONTACT_COOLDOWN,
    "hand_wave": HAND_WAVE_COOLDOWN,
}

ARBITRATION_COOLDOWNS = {
    "greet": GREET_ACTION_COOLDOWN,
    "wave": WAVE_ACTION_COOLDOWN,
}


class CooldownTracker:
    def __init__(self, durations=None):
        self._expires_at = {}
        self._durations = dict(EVENT_COOLDOWNS if durations is None else durations)

    def is_ready(self, event_type, now):
        return float(now) >= self._expires_at.get(event_type, 0.0)

    def remaining(self, event_type, now):
        return max(0.0, self._expires_at.get(event_type, 0.0) - float(now))

    def trigger(self, event_type, now):
        duration = self._durations.get(event_type, 0.0)
        self._expires_at[event_type] = float(now) + duration
        return duration

    def clear(self):
        self._expires_at.clear()
