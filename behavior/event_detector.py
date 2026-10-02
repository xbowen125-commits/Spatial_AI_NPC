"""把逐帧 Player State 稳定地转换成低频 World Event。"""

import time

from events.world_event import WorldEvent


PLAYER_ENTER_CONFIRM_FRAMES = 3
EYE_CONTACT_CONFIRM_FRAMES = 3
HAND_RAISE_CONFIRM_FRAMES = 3
DISTANCE_CONFIRM_FRAMES = 4
PLAYER_LEAVE_SECONDS = 5.0
EYE_CONTACT_LONG_SECONDS = 10.0


class EventDetector:
    def __init__(self):
        self.person_present = False
        self._person_true_frames = 0
        self._person_false_since = None

        self.eye_contact_active = False
        self._eye_true_frames = 0
        self._eye_false_frames = 0
        self._eye_started_at = None
        self._eye_long_emitted = False

        self._stable_hands = {"left": "down", "right": "down"}
        self._hand_candidates = {"left": None, "right": None}
        self._hand_frames = {"left": 0, "right": 0}

        self._stable_distance = "unknown"
        self._distance_candidate = None
        self._distance_frames = 0

    def update(self, player_state, now=None, timestamp=None):
        now = time.monotonic() if now is None else float(now)
        timestamp = (
            int(time.time() * 1000) if timestamp is None else int(timestamp)
        )
        events = []
        context = summarize_player_state(player_state)

        events.extend(self._detect_person(player_state, context, now, timestamp))
        events.extend(self._detect_eye_contact(player_state, context, now, timestamp))
        events.extend(self._detect_hands(player_state, context, timestamp))
        events.extend(self._detect_distance(player_state, context, timestamp))
        return events

    def _detect_person(self, state, context, now, timestamp):
        events = []
        if bool(state.person_detected):
            self._person_false_since = None
            self._person_true_frames += 1
            if (
                not self.person_present
                and self._person_true_frames >= PLAYER_ENTER_CONFIRM_FRAMES
            ):
                self.person_present = True
                events.append(WorldEvent.create("player_enter", context, timestamp))
        else:
            self._person_true_frames = 0
            if self.person_present:
                if self._person_false_since is None:
                    self._person_false_since = now
                elif now - self._person_false_since >= PLAYER_LEAVE_SECONDS:
                    self.person_present = False
                    self._person_false_since = None
                    events.append(WorldEvent.create("player_leave", context, timestamp))
        return events

    def _detect_eye_contact(self, state, context, now, timestamp):
        events = []
        value = str(getattr(state, "eye_contact", "unknown"))
        if value == "true":
            self._eye_true_frames += 1
            self._eye_false_frames = 0
            if (
                not self.eye_contact_active
                and self._eye_true_frames >= EYE_CONTACT_CONFIRM_FRAMES
            ):
                self.eye_contact_active = True
                self._eye_started_at = now
                self._eye_long_emitted = False
                events.append(
                    WorldEvent.create("eye_contact_started", context, timestamp)
                )
            if (
                self.eye_contact_active
                and not self._eye_long_emitted
                and self._eye_started_at is not None
                and now - self._eye_started_at >= EYE_CONTACT_LONG_SECONDS
            ):
                self._eye_long_emitted = True
                events.append(WorldEvent.create("eye_contact_long", context, timestamp))
        else:
            self._eye_true_frames = 0
            self._eye_false_frames += 1
            if self._eye_false_frames >= EYE_CONTACT_CONFIRM_FRAMES:
                self.eye_contact_active = False
                self._eye_started_at = None
                self._eye_long_emitted = False
        return events

    def _detect_hands(self, state, context, timestamp):
        events = []
        for side, attribute in (("left", "left_hand"), ("right", "right_hand")):
            value = str(getattr(state, attribute, "unknown"))
            if value == self._stable_hands[side]:
                self._hand_candidates[side] = None
                self._hand_frames[side] = 0
                continue
            if value != self._hand_candidates[side]:
                self._hand_candidates[side] = value
                self._hand_frames[side] = 1
            else:
                self._hand_frames[side] += 1
            if self._hand_frames[side] < HAND_RAISE_CONFIRM_FRAMES:
                continue

            previous = self._stable_hands[side]
            self._stable_hands[side] = value
            self._hand_candidates[side] = None
            self._hand_frames[side] = 0
            if previous == "down" and value == "raised":
                events.append(WorldEvent.create("hand_wave", context, timestamp))
        return events

    def _detect_distance(self, state, context, timestamp):
        value = str(getattr(state, "distance", "unknown"))
        if value == self._stable_distance:
            self._distance_candidate = None
            self._distance_frames = 0
            return []
        if value != self._distance_candidate:
            self._distance_candidate = value
            self._distance_frames = 1
        else:
            self._distance_frames += 1
        if self._distance_frames < DISTANCE_CONFIRM_FRAMES:
            return []

        previous = self._stable_distance
        self._stable_distance = value
        self._distance_candidate = None
        self._distance_frames = 0
        if previous == "far" and value == "near":
            return [WorldEvent.create("player_approach", context, timestamp)]
        if previous in {"near", "medium"} and value == "far":
            return [WorldEvent.create("player_far", context, timestamp)]
        return []


def summarize_player_state(state):
    """只保留触发行为需要的摘要，不复制完整 Player State。"""
    return {
        "person_detected": bool(getattr(state, "person_detected", False)),
        "horizontal_position": str(
            getattr(state, "horizontal_position", "unknown")
        ),
        "distance": str(getattr(state, "distance", "unknown")),
        "left_hand": str(getattr(state, "left_hand", "unknown")),
        "right_hand": str(getattr(state, "right_hand", "unknown")),
        "eye_contact": str(getattr(state, "eye_contact", "unknown")),
        "looking_at_npc": str(
            getattr(state, "looking_at_npc", "unknown")
        ),
    }
