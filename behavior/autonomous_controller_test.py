"""Autonomous Controller 的确定性双通道测试。"""

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.autonomous_controller import AutonomousController
from behavior.behavior_manager import BehaviorManager
from behavior.relationship_behavior import build_relationship_behavior_profile
from emotion.emotion_state import EmotionState
from events.world_event import WorldEvent
from memory.player_profile import PlayerProfile
from npc.conversation_context import ConversationContext
from npc.personality import get_default_personality


def perception(**changes):
    values = {
        "person_detected": False,
        "eye_contact": "false",
        "speech_detected": False,
        "left_hand": "down",
        "right_hand": "down",
    }
    values.update(changes)
    return values


def profile(level):
    return build_relationship_behavior_profile(level)


def event(event_type, timestamp):
    return WorldEvent.create(event_type, {}, timestamp=timestamp)


class BrokenManager:
    def arbitrate(self, **kwargs):
        del kwargs
        raise ValueError("simulated invalid input")


class AutonomousControllerTests(unittest.TestCase):
    def setUp(self):
        self.manager = BehaviorManager()
        self.controller = AutonomousController(self.manager)

    def test_default_remains_idle_without_dispatch(self):
        result = self.controller.tick(now=0.0)
        self.assertEqual(result.attention, "idle")
        self.assertFalse(result.executed)
        self.assertEqual(result.dispatches, ())

    def test_visible_player_selects_look_attention(self):
        result = self.controller.tick(
            perception(person_detected=True),
            profile("stranger"),
            EmotionState(),
            now=0.0,
        )
        self.assertEqual(result.attention, "look_at_player")
        self.assertIn("look_at_player", result.dispatches)

    def test_repeated_look_is_not_dispatched_again(self):
        inputs = perception(person_detected=True, eye_contact="true")
        first = self.controller.tick(
            inputs, profile("friend"), EmotionState(), now=0.0
        )
        repeated = [
            self.controller.tick(
                inputs,
                profile("friend"),
                EmotionState(),
                now=float(index),
            )
            for index in range(1, 101)
        ]
        self.assertTrue(first.executed)
        self.assertTrue(all(not item.executed for item in repeated))

    def test_attention_change_is_dispatched(self):
        self.controller.tick(
            perception(person_detected=True),
            profile("stranger"),
            EmotionState(),
            now=0.0,
        )
        changed = self.controller.tick(
            perception(person_detected=True),
            profile("stranger"),
            EmotionState.create(0.2, 0.3, 0.0, 1.0),
            now=1.0,
        )
        self.assertEqual(changed.attention, "observe")
        self.assertEqual(changed.dispatches, ("observe",))

    def test_wave_executes_alongside_higher_scored_look(self):
        result = self.controller.tick(
            perception(
                person_detected=True,
                eye_contact="true",
                left_hand="raised",
            ),
            profile("friend"),
            EmotionState.create(0.6, 0.4, 0.8, 0.7),
            personality=get_default_personality(),
            latest_world_event=event("hand_wave", 1),
            now=0.0,
        )
        self.assertEqual(result.arbitration.action, "look_at_player")
        self.assertEqual(result.attention, "look_at_player")
        self.assertEqual(result.interaction, "wave")
        self.assertEqual(result.dispatches, ("look_at_player", "wave"))
        self.assertIn("relationship_allows_wave", result.reasons)

    def test_same_wave_event_is_consumed_once(self):
        wave_event = event("hand_wave", 1)
        inputs = perception(person_detected=True, left_hand="raised")
        first = self.controller.tick(
            inputs,
            profile("friend"),
            EmotionState(),
            latest_world_event=wave_event,
            now=0.0,
        )
        second = self.controller.tick(
            inputs,
            profile("friend"),
            EmotionState(),
            latest_world_event=wave_event,
            now=6.0,
        )
        self.assertEqual(first.interaction, "wave")
        self.assertIsNone(second.interaction)
        self.assertFalse(second.executed)

    def test_stranger_cannot_wave(self):
        result = self.controller.tick(
            perception(person_detected=True, left_hand="raised"),
            profile("stranger"),
            EmotionState(),
            latest_world_event=event("hand_wave", 1),
            now=0.0,
        )
        self.assertIsNone(result.interaction)
        self.assertNotIn("wave", result.dispatches)

    def test_greet_event_executes_only_once(self):
        enter_event = event("player_enter", 1)
        inputs = perception(person_detected=True)
        first = self.controller.tick(
            inputs,
            profile("friend"),
            EmotionState(),
            latest_world_event=enter_event,
            now=0.0,
        )
        second = self.controller.tick(
            inputs,
            profile("friend"),
            EmotionState(),
            latest_world_event=enter_event,
            now=61.0,
        )
        self.assertEqual(first.interaction, "greet")
        self.assertIsNone(second.interaction)

    def test_cooldown_blocks_new_wave_event(self):
        inputs = perception(person_detected=True, left_hand="raised")
        first = self.controller.tick(
            inputs,
            profile("friend"),
            EmotionState(),
            latest_world_event=event("hand_wave", 1),
            now=0.0,
        )
        blocked = self.controller.tick(
            inputs,
            profile("friend"),
            EmotionState(),
            latest_world_event=event("hand_wave", 2),
            now=1.0,
        )
        restored = self.controller.tick(
            inputs,
            profile("friend"),
            EmotionState(),
            latest_world_event=event("hand_wave", 3),
            now=5.0,
        )
        self.assertEqual(first.interaction, "wave")
        self.assertIsNone(blocked.interaction)
        self.assertEqual(restored.interaction, "wave")

    def test_active_conversation_blocks_social_event(self):
        result = self.controller.tick(
            perception(person_detected=True, left_hand="raised"),
            profile("friend"),
            EmotionState(),
            conversation_active=True,
            latest_world_event=event("hand_wave", 1),
            now=0.0,
        )
        self.assertIsNone(result.interaction)
        self.assertNotIn("wave", result.dispatches)

    def test_listening_and_speaking_states_have_priority(self):
        self.manager.on_speech_event()
        listening = self.controller.tick(
            perception(person_detected=True, speech_detected=True),
            profile("friend"),
            EmotionState(),
            now=0.0,
        )
        self.assertEqual(listening.interaction, "listen")

        self.manager.on_response_generated()
        speaking = self.controller.tick(
            perception(person_detected=True),
            profile("friend"),
            EmotionState(),
            now=1.0,
        )
        self.assertEqual(speaking.interaction, "speak")

    def test_player_leave_returns_attention_to_idle(self):
        self.controller.tick(
            perception(person_detected=True),
            profile("friend"),
            EmotionState(),
            now=0.0,
        )
        result = self.controller.tick(
            perception(),
            profile("friend"),
            EmotionState(),
            latest_world_event=event("player_leave", 2),
            now=1.0,
        )
        self.assertEqual(result.attention, "idle")
        self.assertEqual(result.dispatches, ("idle",))

    def test_equal_sequences_are_deterministic(self):
        inputs = perception(person_detected=True, eye_contact="true")
        first = AutonomousController(BehaviorManager()).tick(
            inputs, profile("friend"), EmotionState(), now=0.0
        )
        second = AutonomousController(BehaviorManager()).tick(
            inputs, profile("friend"), EmotionState(), now=0.0
        )
        self.assertEqual(first, second)

    def test_decision_contains_explanation(self):
        result = self.controller.tick(
            perception(person_detected=True, eye_contact="true"),
            profile("friend"),
            EmotionState.create(0.5, 0.3, 0.8, 0.7),
            now=0.0,
        )
        self.assertTrue(result.reasons)
        self.assertIn("attention_changed", result.reasons)
        self.assertIsNotNone(result.arbitration)

    def test_personality_is_not_modified(self):
        personality = get_default_personality()
        before = copy.deepcopy(personality.to_dict())
        self.controller.tick(
            perception(person_detected=True),
            profile("friend"),
            EmotionState(),
            personality=personality,
            now=0.0,
        )
        self.assertEqual(personality.to_dict(), before)

    def test_relationship_memory_is_not_modified(self):
        memory = PlayerProfile(relationship_level="friend", times_seen=5)
        before = copy.deepcopy(memory.to_dict())
        self.controller.tick(
            perception(person_detected=True),
            profile(memory.relationship_level),
            EmotionState(),
            now=0.0,
        )
        self.assertEqual(memory.to_dict(), before)

    def test_conversation_context_is_not_modified(self):
        conversation = ConversationContext(session_id="autonomous-test")
        conversation.add_user_turn("你好", timestamp=1)
        before = conversation.get_recent_context()
        self.controller.tick(
            perception(person_detected=True),
            profile("friend"),
            EmotionState(),
            conversation_active=True,
            now=0.0,
        )
        self.assertEqual(conversation.get_recent_context(), before)

    def test_emotion_is_read_only(self):
        emotion = EmotionState.create(0.6, 0.4, 0.8, 0.7)
        before = emotion.snapshot()
        self.controller.tick(
            perception(person_detected=True),
            profile("friend"),
            emotion,
            now=0.0,
        )
        self.assertEqual(emotion.snapshot(), before)

    def test_invalid_module_input_falls_back_safely(self):
        controller = AutonomousController(BrokenManager())
        result = controller.tick(perception=object(), now=0.0)
        self.assertEqual(result.attention, "idle")
        self.assertTrue(result.fallback)
        self.assertFalse(result.executed)
        self.assertIn("arbitration_error", result.reasons)

    def test_no_event_means_no_discrete_action(self):
        result = self.controller.tick(
            perception(person_detected=True, left_hand="raised"),
            profile("friend"),
            EmotionState(),
            now=0.0,
        )
        self.assertIsNone(result.interaction)
        self.assertNotIn("wave", result.dispatches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
