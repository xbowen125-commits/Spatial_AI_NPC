"""Runtime 调度测试，使用临时 Memory 和 Fake 模块，无外部 API。"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.npc_runtime import NPCManager
from emotion.emotion_engine import EmotionEngine
from events.world_event import WorldEvent
from memory.memory_store import MemoryStore


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = NPCManager(emotion_engine=EmotionEngine(now=0))
        self.visible = {"person_detected": True, "eye_contact": "true"}

    def tick(self, event=None, **kwargs):
        return self.runtime.tick(
            player_state=self.visible,
            world_events=(() if event is None else (event,)),
            relationship_state="friend", now=0, **kwargs)

    def test_tick(self):
        self.assertEqual(self.runtime.tick(now=0).dispatch_actions, ())

    def test_enter(self):
        result = self.tick(WorldEvent.create("player_enter", {}, timestamp=1))
        self.assertEqual(result.autonomous_decision.attention, "look_at_player")

    def test_wave_with_look(self):
        result = self.tick(WorldEvent.create("hand_wave", {}, timestamp=1))
        self.assertEqual(result.autonomous_decision.attention, "look_at_player")
        self.assertEqual(result.autonomous_decision.interaction, "wave")

    def test_leave(self):
        self.tick()
        result = self.runtime.tick(player_state={"person_detected": False},
            world_events=[WorldEvent.create("player_leave", {})], now=0)
        self.assertEqual(result.autonomous_decision.attention, "idle")
        self.assertEqual(result.dispatch_actions, ("idle",))

    def test_conversation_gate(self):
        result = self.tick(WorldEvent.create("hand_wave", {}),
                           conversation_state={"active": True})
        self.assertIsNone(result.autonomous_decision.interaction)

    def test_emotion_update_once(self):
        event = WorldEvent.create("hand_wave", {})
        first = self.tick(event)
        second = self.tick(event)
        self.assertEqual(first.emotion_snapshot, second.emotion_snapshot)
        self.assertAlmostEqual(first.emotion_snapshot["valence"], .25)
        self.assertEqual(second.events_processed, ())

    def test_relationship_update(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "profile.json")
            self.runtime = NPCManager(memory_store=store,
                                      emotion_engine=EmotionEngine(now=0))
            result = self.tick(WorldEvent.create("player_enter", {}, timestamp=1))
            self.assertEqual(result.relationship_update["times_seen"], 1)

    def test_controller_integration(self):
        first = self.tick()
        second = self.tick()
        self.assertEqual(first.dispatch_actions, ("look_at_player",))
        self.assertEqual(second.dispatch_actions, ())

    def test_module_failure(self):
        self.runtime.emotion_engine = Mock()
        self.runtime.emotion_engine.update.side_effect = RuntimeError()
        self.assertEqual(self.tick().dispatch_actions, ("idle",))

    def test_invalid_input(self):
        result = self.tick(object())
        self.assertEqual(result.dispatch_actions, ("idle",))
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
