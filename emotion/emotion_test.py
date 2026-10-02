"""无需Camera、Microphone、Unity或LLM的Emotion System测试。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion.emotion_engine import EmotionEngine
from emotion.emotion_policy import (
    EmotionAwareBehaviorRules,
    EmotionStimulus,
)
from emotion.emotion_state import EmotionState
from evaluation.interaction_logger import EmotionLogger
from events.interaction_event import InteractionEvent, NpcState
from events.world_event import WorldEvent
from memory.player_profile import PlayerProfile
from npc.world_context import build_world_context


def world_event(event_type):
    return WorldEvent.create(event_type, {})


class EmotionTests(unittest.TestCase):
    def test_initial_state_is_neutral(self):
        state = EmotionEngine(now=0.0).state
        self.assertEqual(state.emotion, "neutral")
        self.assertEqual(state.intensity, 0.0)

    def test_player_enter_is_happy(self):
        engine = EmotionEngine(now=0.0)
        state = engine.process_world_event(
            world_event("player_enter"),
            "acquaintance",
            now=0.0,
        )
        self.assertEqual(state.emotion, "happy")
        self.assertAlmostEqual(state.intensity, 0.45)

    def test_hand_wave_strengthens_happy(self):
        engine = EmotionEngine(now=0.0)
        first = engine.process_world_event(
            world_event("player_enter"), "acquaintance", now=0.0
        )
        waved = engine.process_world_event(
            world_event("hand_wave"), "acquaintance", now=0.1
        )
        self.assertEqual(waved.emotion, "happy")
        self.assertGreater(waved.intensity, first.intensity)

    def test_eye_contact_long_is_curious(self):
        engine = EmotionEngine(now=0.0)
        state = engine.process_world_event(
            world_event("eye_contact_long"),
            "acquaintance",
            now=0.0,
        )
        self.assertEqual(state.emotion, "curious")

    def test_friend_slightly_increases_intensity(self):
        friend = EmotionEngine(now=0.0).process_world_event(
            world_event("player_enter"), "friend", now=0.0
        )
        stranger = EmotionEngine(now=0.0).process_world_event(
            world_event("player_enter"), "stranger", now=0.0
        )
        self.assertGreater(friend.intensity, stranger.intensity)
        self.assertAlmostEqual(friend.intensity, 0.45 * 1.15)

    def test_decay_returns_to_neutral(self):
        engine = EmotionEngine(now=0.0)
        engine.process_world_event(
            world_event("hand_wave"), "acquaintance", now=0.0
        )
        state = engine.update(now=30.0, force=True)
        self.assertEqual(state.emotion, "neutral")
        self.assertEqual(state.cause, "none")
        self.assertEqual(state.intensity, 0.0)

    def test_active_emotions_blend(self):
        engine = EmotionEngine(now=0.0)
        engine.process_world_event(
            world_event("player_enter"), "acquaintance", now=0.0
        )
        state = engine.process_world_event(
            world_event("hand_wave"), "acquaintance", now=0.0
        )
        self.assertAlmostEqual(state.intensity, 0.45 * 0.35 + 0.60 * 0.65)

    def test_all_values_are_clamped(self):
        engine = EmotionEngine(now=0.0)
        state = engine.apply_stimulus(
            EmotionStimulus("happy", 4.0, 3.0, 2.0, "test"),
            "friend",
            now=0.0,
        )
        self.assertEqual(state.valence, 1.0)
        self.assertEqual(state.arousal, 1.0)
        self.assertEqual(state.intensity, 1.0)

    def test_new_engine_does_not_restore_old_emotion(self):
        first = EmotionEngine(now=0.0)
        first.process_world_event(world_event("hand_wave"), now=0.0)
        second = EmotionEngine(now=0.0)
        self.assertEqual(second.state.emotion, "neutral")

    def test_memory_profile_has_no_emotion_fields(self):
        fields = set(PlayerProfile().to_dict())
        self.assertNotIn("emotion", fields)
        self.assertNotIn("valence", fields)
        self.assertNotIn("arousal", fields)
        self.assertNotIn("intensity", fields)

    def test_speech_only_boosts_arousal(self):
        engine = EmotionEngine(now=0.0)
        before = engine.state
        after = engine.process_speech_event("friend", now=0.0)
        self.assertEqual(after.emotion, before.emotion)
        self.assertEqual(after.intensity, before.intensity)
        self.assertAlmostEqual(after.arousal, before.arousal + 0.05)

    def test_emotion_does_not_create_world_event(self):
        result = EmotionEngine(now=0.0).process_world_event(
            world_event("player_enter"), now=0.0
        )
        self.assertIsInstance(result, EmotionState)
        self.assertFalse(hasattr(result, "event_type"))

    def test_behavior_read_does_not_feed_back_into_emotion(self):
        engine = EmotionEngine(now=0.0)
        engine.process_world_event(
            world_event("eye_contact_long"), "acquaintance", now=0.0
        )
        before = engine.state.to_dict()
        rules = EmotionAwareBehaviorRules(lambda: engine.state)
        decision = rules.decide(world_event("eye_contact_long"))
        self.assertEqual(decision.action, "look_at_player")
        self.assertEqual(engine.state.to_dict(), before)

    def test_npc_state_schema_and_agent_context(self):
        state = NpcState.create(False, False, "happy", "idle", 0.42)
        self.assertAlmostEqual(state.to_dict()["npc_emotion_intensity"], 0.42)
        player = SimpleNamespace(
            person_detected=True,
            face_detected=True,
            horizontal_position="center",
            eye_contact="true",
            looking_at_npc="true",
            distance="medium",
            left_hand="down",
            right_hand="down",
            directed_speech="true",
        )
        speech = SimpleNamespace(text="你好", language="zh", confidence=None)
        event = InteractionEvent.speech(
            speech,
            player,
            "friend",
            "happy",
            0.42,
        )
        context = build_world_context(event)
        self.assertIn("NPC current emotion is happy.", context)
        self.assertIn("NPC emotion intensity is 0.42.", context)
        transport = event.to_dict()["context"]
        self.assertNotIn("npc_emotion", transport)
        self.assertNotIn("npc_emotion_intensity", transport)

    def test_emotion_log_contains_no_speech_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emotions.jsonl"
            engine = EmotionEngine(logger=EmotionLogger(path), now=0.0)
            engine.process_speech_event("friend", now=0.0)
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["trigger_event"], "speech_event")
        self.assertNotIn("speech_text", row)
        self.assertNotIn("text", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
