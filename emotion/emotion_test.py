"""连续 Dynamic Emotion State 与 Agent 独立上下文测试。"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.speech_recognition import SpeechResult
from emotion.emotion_engine import EmotionEngine
from emotion.emotion_policy import EVENT_DELTAS, EmotionAwareBehaviorRules
from emotion.emotion_state import EmotionState
from evaluation.interaction_logger import EmotionLogger
from events.interaction_event import InteractionEvent
from events.world_event import WorldEvent
from memory.memory_policy import MemoryCandidate
from memory.memory_store import MemoryStore
from npc.agent import NpcAgent
from npc.conversation_context import ConversationContext
from npc.personality import get_default_personality


def world_event(event_type):
    return WorldEvent.create(event_type, {})


def speech_event(text="你好", relationship_level="friend"):
    player = SimpleNamespace(
        person_detected=True,
        face_detected=True,
        horizontal_position="center",
        distance="medium",
        left_hand="down",
        right_hand="down",
        eye_contact="true",
        looking_at_npc="true",
        directed_speech="true",
    )
    return InteractionEvent.speech(
        SpeechResult(text, "zh"),
        player,
        relationship_level=relationship_level,
    )


class FakeProvider:
    def __init__(self):
        self.messages = None

    def generate(self, messages):
        self.messages = list(messages)
        return json.dumps(
            {
                "reply": "好的。",
                "action": "none",
                "emotion": "neutral",
            },
            ensure_ascii=False,
        )


class EmotionStateTests(unittest.TestCase):
    def test_default_state_matches_baseline(self):
        state = EmotionState()
        self.assertEqual(state.valence, 0.20)
        self.assertEqual(state.arousal, 0.30)
        self.assertEqual(state.social_comfort, 0.40)
        self.assertEqual(state.curiosity, 0.50)
        self.assertEqual(state.intensity, 0.0)

    def test_all_fields_are_in_valid_ranges(self):
        state = EmotionState.create(9.0, -3.0, 4.0, -2.0)
        self.assertTrue(-1.0 <= state.valence <= 1.0)
        self.assertTrue(0.0 <= state.arousal <= 1.0)
        self.assertTrue(0.0 <= state.social_comfort <= 1.0)
        self.assertTrue(0.0 <= state.curiosity <= 1.0)

    def test_friendly_conversation_delta(self):
        state = EmotionEngine(now=0.0).apply_event(
            "friendly_conversation",
            now=0.0,
        )
        self.assertAlmostEqual(state.valence, 0.30)
        self.assertAlmostEqual(state.arousal, 0.32)
        self.assertAlmostEqual(state.social_comfort, 0.48)
        self.assertAlmostEqual(state.curiosity, 0.52)

    def test_negative_interaction_delta(self):
        state = EmotionEngine(now=0.0).apply_event(
            "negative_interaction",
            now=0.0,
        )
        self.assertAlmostEqual(state.valence, 0.05)
        self.assertAlmostEqual(state.arousal, 0.40)
        self.assertAlmostEqual(state.social_comfort, 0.30)
        self.assertAlmostEqual(state.curiosity, 0.48)

    def test_eye_contact_delta(self):
        state = EmotionEngine(now=0.0).apply_event("eye_contact", now=0.0)
        self.assertAlmostEqual(state.valence, 0.20)
        self.assertAlmostEqual(state.arousal, 0.32)
        self.assertAlmostEqual(state.social_comfort, 0.43)
        self.assertAlmostEqual(state.curiosity, 0.52)

    def test_player_wave_delta(self):
        state = EmotionEngine(now=0.0).apply_event("player_wave", now=0.0)
        self.assertEqual(
            state,
            EmotionState.create(0.25, 0.35, 0.44, 0.51),
        )

    def test_upper_clamp(self):
        engine = EmotionEngine(now=0.0)
        for _ in range(30):
            engine.apply_event("friendly_conversation", now=0.0)
            engine.apply_event("player_wave", now=0.0)
        self.assertEqual(engine.state.valence, 1.0)
        self.assertEqual(engine.state.arousal, 1.0)
        self.assertEqual(engine.state.social_comfort, 1.0)
        self.assertEqual(engine.state.curiosity, 1.0)

    def test_lower_clamp(self):
        state = EmotionState.create(-9.0, -3.0, -4.0, -2.0)
        self.assertEqual(state.valence, -1.0)
        self.assertEqual(state.arousal, 0.0)
        self.assertEqual(state.social_comfort, 0.0)
        self.assertEqual(state.curiosity, 0.0)

    def test_decay_moves_toward_baseline(self):
        engine = EmotionEngine(now=0.0)
        engine.apply_event("friendly_conversation", now=0.0)
        state = engine.decay_toward_baseline(step=0.05, now=1.0)
        self.assertAlmostEqual(state.valence, 0.25)
        self.assertAlmostEqual(state.arousal, 0.30)
        self.assertAlmostEqual(state.social_comfort, 0.43)
        self.assertAlmostEqual(state.curiosity, 0.50)

    def test_decay_does_not_cross_baseline(self):
        state = EmotionState.create(0.22, 0.28, 0.42, 0.48)
        decayed = state.decay_toward_baseline(step=0.05)
        self.assertEqual(decayed, EmotionState())

    def test_labels_are_derived_from_continuous_values(self):
        self.assertEqual(EmotionState().labels(), ("calm",))
        warm = EmotionState.create(0.50, 0.30, 0.80, 0.80)
        self.assertEqual(warm.labels(), ("calm", "warm", "curious"))
        self.assertIn("uneasy", EmotionState.create(-0.20, 0.4, 0.2, 0.5).labels())
        self.assertIn("excited", EmotionState.create(0.4, 0.8, 0.5, 0.5).labels())

    def test_context_is_deterministic(self):
        state = EmotionState.create(0.55, 0.30, 0.80, 0.72)
        self.assertEqual(state.to_context(), state.to_context())
        self.assertIn("EMOTION CONTEXT:", state.to_context())
        self.assertIn("valence=0.55", state.to_context())
        self.assertIn("labels=calm,warm,curious", state.to_context())

    def test_repeated_events_are_predictable(self):
        first = EmotionEngine(now=0.0)
        second = EmotionEngine(now=0.0)
        for _ in range(3):
            first.apply_event("friendly_conversation", now=0.0)
            second.apply_event("friendly_conversation", now=0.0)
        self.assertEqual(first.state, second.state)
        self.assertAlmostEqual(first.state.valence, 0.50)
        self.assertAlmostEqual(first.state.arousal, 0.36)
        self.assertAlmostEqual(first.state.social_comfort, 0.64)
        self.assertAlmostEqual(first.state.curiosity, 0.56)

    def test_all_supported_events_have_fixed_deltas(self):
        self.assertEqual(
            set(EVENT_DELTAS),
            {
                "friendly_conversation",
                "negative_interaction",
                "eye_contact",
                "player_wave",
                "player_absent",
                "long_idle",
            },
        )

    def test_personality_is_not_modified(self):
        profile = get_default_personality()
        before = copy.deepcopy(profile.to_dict())
        EmotionEngine(now=0.0).apply_event("negative_interaction", now=0.0)
        self.assertEqual(profile.to_dict(), before)

    def test_relationship_memory_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player_profile.json"
            store = MemoryStore(path)
            store.remember(MemoryCandidate("interaction", "seen", timestamp=1))
            before = path.read_bytes()
            EmotionEngine(now=0.0).apply_event("player_wave", now=0.0)
            self.assertEqual(path.read_bytes(), before)

    def test_conversation_context_is_not_modified(self):
        conversation = ConversationContext(session_id="emotion-test")
        conversation.add_user_turn("你好", timestamp=1)
        before = conversation.get_recent_context()
        EmotionEngine(now=0.0).apply_event("eye_contact", now=0.0)
        self.assertEqual(conversation.get_recent_context(), before)

    def test_agent_receives_separate_emotion_context(self):
        engine = EmotionEngine(now=0.0)
        engine.apply_event("friendly_conversation", now=0.0)
        provider = FakeProvider()
        agent = NpcAgent(
            provider=provider,
            llm_enabled=True,
            emotion_state_provider=lambda: engine.state,
        )
        result = agent._make_decision(speech_event())
        self.assertEqual(result.decision.reply, "好的。")
        emotion_messages = [
            item for item in provider.messages
            if item["role"] == "system"
            and item["content"].startswith("EMOTION CONTEXT:")
        ]
        self.assertEqual(len(emotion_messages), 1)
        self.assertIn("social_comfort=0.48", emotion_messages[0]["content"])
        self.assertTrue(
            any(
                item["content"].startswith("PERSONALITY CONTEXT:")
                for item in provider.messages
                if item["role"] == "system"
            )
        )
        self.assertTrue(
            any(
                item["content"].startswith("RELATIONSHIP BEHAVIOR CONTEXT:")
                for item in provider.messages
                if item["role"] == "system"
            )
        )

    def test_new_engine_resets_to_baseline(self):
        first = EmotionEngine(now=0.0)
        first.apply_event("friendly_conversation", now=0.0)
        second = EmotionEngine(now=0.0)
        self.assertEqual(second.state, EmotionState())

    def test_world_event_and_behavior_compatibility(self):
        engine = EmotionEngine(now=0.0)
        state = engine.process_world_event(
            world_event("eye_contact_long"),
            now=0.0,
        )
        self.assertIn("curious", state.labels())
        rules = EmotionAwareBehaviorRules(lambda: engine.state)
        decision = rules.decide(world_event("eye_contact_long"))
        self.assertEqual(decision.action, "look_at_player")

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
