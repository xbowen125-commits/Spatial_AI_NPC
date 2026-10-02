"""无需 Camera、Microphone、Unity 或 LLM 的主动行为测试。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.behavior_manager import BehaviorManager
from behavior.event_detector import EventDetector
from evaluation.interaction_logger import BehaviorLogger
from events.interaction_event import NpcState
from events.world_event import WorldEvent


def state(**changes):
    values = {
        "person_detected": False,
        "horizontal_position": "unknown",
        "distance": "unknown",
        "left_hand": "down",
        "right_hand": "down",
        "eye_contact": "false",
        "looking_at_npc": "false",
    }
    values.update(changes)
    return SimpleNamespace(**values)


class MemoryLogger:
    def __init__(self):
        self.rows = []

    def write(self, event, decision, status, **details):
        self.rows.append((event.event_type, decision, status, details))


class EventDetectorTests(unittest.TestCase):
    def test_player_enter_once_after_three_frames(self):
        detector = EventDetector()
        events = []
        for frame in range(8):
            events.extend(
                detector.update(
                    state(person_detected=True),
                    now=frame * 0.1,
                    timestamp=frame,
                )
            )
        self.assertEqual(
            [event.event_type for event in events].count("player_enter"),
            1,
        )

    def test_player_leave_after_five_seconds(self):
        detector = EventDetector()
        for frame in range(3):
            detector.update(state(person_detected=True), now=frame * 0.1)
        self.assertFalse(detector.update(state(), now=1.0))
        events = detector.update(state(), now=6.1)
        self.assertEqual([event.event_type for event in events], ["player_leave"])

    def test_eye_contact_long(self):
        detector = EventDetector()
        events = []
        looking = state(
            person_detected=True,
            eye_contact="true",
            looking_at_npc="true",
        )
        for now in (0.0, 0.1, 0.2):
            events.extend(detector.update(looking, now=now))
        events.extend(detector.update(looking, now=10.3))
        names = [event.event_type for event in events]
        self.assertEqual(names.count("eye_contact_started"), 1)
        self.assertEqual(names.count("eye_contact_long"), 1)

    def test_hand_wave_after_three_frames(self):
        detector = EventDetector()
        events = []
        for frame in range(3):
            events.extend(
                detector.update(
                    state(person_detected=True, left_hand="raised"),
                    now=frame * 0.1,
                )
            )
        self.assertIn("hand_wave", [event.event_type for event in events])

    def test_far_to_near_after_four_frames(self):
        detector = EventDetector()
        for frame in range(4):
            detector.update(state(distance="far"), now=frame * 0.1)
        events = []
        for frame in range(4):
            events.extend(
                detector.update(state(distance="near"), now=1 + frame * 0.1)
            )
        self.assertEqual(
            [event.event_type for event in events],
            ["player_approach"],
        )
        far_events = []
        for frame in range(4):
            far_events.extend(
                detector.update(state(distance="far"), now=2 + frame * 0.1)
            )
        self.assertEqual(
            [event.event_type for event in far_events],
            ["player_far"],
        )


class BehaviorManagerTests(unittest.TestCase):
    def test_priority_selects_player_enter(self):
        logger = MemoryLogger()
        manager = BehaviorManager(logger=logger)
        decision = manager.handle_events(
            [
                WorldEvent.create("hand_wave", {}),
                WorldEvent.create("player_enter", {}),
            ],
            now=0.0,
        )
        self.assertEqual(decision.reply, "欢迎回来。")
        self.assertEqual(decision.action, "wave")
        self.assertEqual(manager.state, "greeting")
        self.assertIn("superseded", [row[2] for row in logger.rows])

    def test_hand_wave_cooldown(self):
        logger = MemoryLogger()
        manager = BehaviorManager(logger=logger)
        event = WorldEvent.create("hand_wave", {})
        self.assertIsNotNone(manager.handle_events([event], now=0.0))
        manager.on_tts_finished(now=0.0)
        manager.tick(now=0.6)
        self.assertIsNone(manager.handle_events([event], now=2.0))
        self.assertIn("cooldown_blocked", [row[2] for row in logger.rows])

    def test_speaking_queues_low_priority(self):
        manager = BehaviorManager()
        manager.on_agent_started()
        event = WorldEvent.create("eye_contact_long", {})
        self.assertIsNone(manager.handle_events([event], now=0.0))
        self.assertEqual(manager.state, "thinking")
        manager.on_response_generated()
        manager.on_tts_finished(now=1.0)
        decision = manager.tick(now=1.6)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.reply, "有什么想问我的吗？")

    def test_speech_cancels_pending_behavior(self):
        logger = MemoryLogger()
        manager = BehaviorManager(logger=logger)
        manager.on_agent_started()
        manager.handle_events(
            [WorldEvent.create("eye_contact_long", {})],
            now=0.0,
        )
        manager.on_speech_event()
        self.assertEqual(manager.state, "listening")
        self.assertIn("cancelled_by_speech", [row[2] for row in logger.rows])

    def test_behavior_log_and_npc_state_schema(self):
        event = WorldEvent.create(
            "hand_wave",
            {"left_hand": "raised", "right_hand": "down"},
            timestamp=123,
        )
        manager = BehaviorManager()
        decision = manager.handle_events([event], now=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.jsonl"
            BehaviorLogger(path).write(
                event,
                decision,
                "executed",
                behavior_state=manager.state,
            )
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["event_type"], "hand_wave")
        self.assertEqual(row["action"], "wave")
        self.assertEqual(row["cooldown_status"], "executed")
        npc_state = NpcState.create(False, True, "happy", "greeting")
        self.assertEqual(npc_state.to_dict()["npc_behavior_state"], "greeting")


if __name__ == "__main__":
    unittest.main(verbosity=2)
