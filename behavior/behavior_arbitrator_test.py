"""无需 Camera、Unity 或 LLM 的 Behavior Arbitration 测试。"""

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.behavior_arbitrator import BehaviorArbitrator, select_highest
from behavior.behavior_candidate import BehaviorCandidate
from behavior.behavior_manager import BehaviorManager
from behavior.relationship_behavior import build_relationship_behavior_profile
from emotion.emotion_state import EmotionState
from memory.player_profile import PlayerProfile
from npc.conversation_context import ConversationContext
from npc.personality import get_default_personality


def perception(**changes):
    values = {
        "person_detected": False,
        "eye_contact": "false",
        "speech_detected": False,
        "player_entered": False,
        "player_waving": False,
        "left_hand": "down",
        "right_hand": "down",
    }
    values.update(changes)
    return values


def profile(level):
    return build_relationship_behavior_profile(level)


def candidate(decision, action):
    return next(item for item in decision.candidates if item.action == action)


class BehaviorArbitratorTests(unittest.TestCase):
    def test_default_is_idle(self):
        decision = BehaviorArbitrator().select(now=0.0)
        self.assertEqual(decision.action, "idle")

    def test_visible_player_selects_look_or_observe(self):
        decision = BehaviorArbitrator().select(
            perception(person_detected=True),
            profile("stranger"),
            EmotionState(),
            now=0.0,
        )
        self.assertIn(decision.action, {"look_at_player", "observe"})

    def test_eye_contact_increases_look_score(self):
        without_eye = BehaviorArbitrator().select(
            perception(person_detected=True),
            profile("stranger"),
            EmotionState(),
            now=0.0,
        )
        with_eye = BehaviorArbitrator().select(
            perception(person_detected=True, eye_contact="true"),
            profile("stranger"),
            EmotionState(),
            now=0.0,
        )
        self.assertGreater(
            candidate(with_eye, "look_at_player").score,
            candidate(without_eye, "look_at_player").score,
        )

    def test_friend_allows_proactive_wave(self):
        decision = BehaviorArbitrator().select(
            perception(person_detected=True, player_waving=True),
            profile("friend"),
            EmotionState(),
            event_type="hand_wave",
            now=0.0,
        )
        self.assertEqual(decision.action, "wave")
        self.assertTrue(candidate(decision, "wave").eligible)

    def test_stranger_blocks_proactive_wave(self):
        decision = BehaviorArbitrator().select(
            perception(person_detected=True, player_waving=True),
            profile("stranger"),
            EmotionState(),
            event_type="hand_wave",
            now=0.0,
        )
        wave = candidate(decision, "wave")
        self.assertFalse(wave.eligible)
        self.assertIn("relationship_blocks_wave", wave.reasons)
        self.assertNotEqual(decision.action, "wave")

    def test_relationship_greeting_gate(self):
        visible_entry = perception(person_detected=True, player_entered=True)
        stranger = BehaviorArbitrator().select(
            visible_entry,
            profile("stranger"),
            EmotionState(),
            event_type="player_enter",
            now=0.0,
        )
        friend = BehaviorArbitrator().select(
            visible_entry,
            profile("friend"),
            EmotionState(),
            event_type="player_enter",
            now=0.0,
        )
        self.assertFalse(candidate(stranger, "greet").eligible)
        self.assertEqual(friend.action, "greet")

    def test_listening_has_highest_priority(self):
        decision = BehaviorArbitrator().select(
            perception(
                person_detected=True,
                eye_contact="true",
                speech_detected=True,
                player_entered=True,
            ),
            profile("friend"),
            EmotionState.create(0.8, 0.8, 0.9, 0.9),
            now=0.0,
        )
        self.assertEqual(decision.action, "listen")

    def test_speaking_does_not_repeat_greeting(self):
        decision = BehaviorArbitrator().select(
            perception(person_detected=True, player_entered=True),
            profile("friend"),
            EmotionState(),
            behavior_state="speaking",
            event_type="player_enter",
            now=0.0,
        )
        self.assertEqual(decision.action, "speak")
        self.assertFalse(candidate(decision, "greet").eligible)
        self.assertIn("npc_busy", candidate(decision, "greet").reasons)

    def test_positive_emotion_boosts_social_score(self):
        inputs = perception(person_detected=True, player_entered=True)
        neutral = BehaviorArbitrator().select(
            inputs,
            profile("friend"),
            EmotionState(),
            event_type="player_enter",
            now=0.0,
        )
        positive = BehaviorArbitrator().select(
            inputs,
            profile("friend"),
            EmotionState.create(0.8, 0.3, 0.8, 0.5),
            event_type="player_enter",
            now=0.0,
        )
        self.assertGreater(
            candidate(positive, "greet").score,
            candidate(neutral, "greet").score,
        )

    def test_low_comfort_reduces_social_score(self):
        inputs = perception(person_detected=True, player_waving=True)
        low = BehaviorArbitrator().select(
            inputs,
            profile("friend"),
            EmotionState.create(0.4, 0.3, 0.1, 0.5),
            event_type="hand_wave",
            now=0.0,
        )
        high = BehaviorArbitrator().select(
            inputs,
            profile("friend"),
            EmotionState.create(0.4, 0.3, 0.9, 0.5),
            event_type="hand_wave",
            now=0.0,
        )
        self.assertLess(
            candidate(low, "wave").score,
            candidate(high, "wave").score,
        )
        self.assertIn("low_social_comfort", candidate(low, "wave").reasons)

    def test_curiosity_increases_observe_score(self):
        low = BehaviorArbitrator().select(
            perception(person_detected=True),
            profile("stranger"),
            EmotionState.create(0.2, 0.3, 0.4, 0.1),
            now=0.0,
        )
        high = BehaviorArbitrator().select(
            perception(person_detected=True),
            profile("stranger"),
            EmotionState.create(0.2, 0.3, 0.4, 0.9),
            now=0.0,
        )
        self.assertGreater(
            candidate(high, "observe").score,
            candidate(low, "observe").score,
        )

    def test_cooldown_prevents_repeated_wave(self):
        arbitrator = BehaviorArbitrator()
        inputs = perception(person_detected=True, player_waving=True)
        first = arbitrator.select(
            inputs,
            profile("friend"),
            EmotionState(),
            event_type="hand_wave",
            now=0.0,
        )
        repeated = arbitrator.select(
            inputs,
            profile("friend"),
            EmotionState(),
            event_type="hand_wave",
            now=1.0,
        )
        self.assertEqual(first.action, "wave")
        self.assertNotEqual(repeated.action, "wave")
        self.assertIn("wave_cooldown_active", candidate(repeated, "wave").reasons)

    def test_cooldown_expiry_restores_wave(self):
        arbitrator = BehaviorArbitrator()
        inputs = perception(person_detected=True, player_waving=True)
        arbitrator.select(
            inputs,
            profile("friend"),
            EmotionState(),
            event_type="hand_wave",
            now=0.0,
        )
        decision = arbitrator.select(
            inputs,
            profile("friend"),
            EmotionState(),
            event_type="hand_wave",
            now=5.0,
        )
        self.assertEqual(decision.action, "wave")

    def test_tie_breaking_is_fixed(self):
        selected = select_highest(
            (
                BehaviorCandidate("wave", 0.50, ("tie",)),
                BehaviorCandidate("look_at_player", 0.50, ("tie",)),
                BehaviorCandidate("listen", 0.50, ("tie",)),
            )
        )
        self.assertEqual(selected.action, "listen")

    def test_same_input_has_same_result(self):
        arbitrator = BehaviorArbitrator()
        inputs = perception(person_detected=True, eye_contact="true")
        first = arbitrator.select(
            inputs, profile("friend"), EmotionState(), now=0.0
        )
        second = arbitrator.select(
            inputs, profile("friend"), EmotionState(), now=0.0
        )
        self.assertEqual(first, second)

    def test_decision_explains_selection(self):
        decision = BehaviorArbitrator().select(
            perception(person_detected=True, eye_contact="true"),
            profile("friend"),
            EmotionState.create(0.6, 0.4, 0.8, 0.7),
            now=0.0,
        )
        self.assertEqual(decision.action, "look_at_player")
        self.assertIn("player_visible", decision.reasons)
        self.assertIn("eye_contact", decision.reasons)
        self.assertIn("high_social_comfort", decision.reasons)
        self.assertGreater(len(decision.candidates), 1)

    def test_does_not_modify_emotion(self):
        emotion = EmotionState.create(0.6, 0.4, 0.8, 0.7)
        before = emotion.snapshot()
        BehaviorArbitrator().select(
            perception(person_detected=True),
            profile("friend"),
            emotion,
            now=0.0,
        )
        self.assertEqual(emotion.snapshot(), before)

    def test_does_not_modify_relationship_memory(self):
        memory = PlayerProfile(relationship_level="friend", times_seen=5)
        before = copy.deepcopy(memory.to_dict())
        BehaviorArbitrator().select(
            perception(person_detected=True),
            profile(memory.relationship_level),
            EmotionState(),
            now=0.0,
        )
        self.assertEqual(memory.to_dict(), before)

    def test_does_not_modify_conversation_context(self):
        conversation = ConversationContext(session_id="arbitration-test")
        conversation.add_user_turn("你好", timestamp=1)
        before = conversation.get_recent_context()
        BehaviorArbitrator().select(
            perception(person_detected=True),
            profile("friend"),
            EmotionState(),
            conversation_active=True,
            now=0.0,
        )
        self.assertEqual(conversation.get_recent_context(), before)

    def test_manager_exposes_arbitration_without_executing(self):
        manager = BehaviorManager()
        decision = manager.arbitrate(
            perception(person_detected=True, eye_contact="true"),
            profile("friend"),
            EmotionState(),
            personality=get_default_personality(),
            now=0.0,
        )
        self.assertEqual(decision.action, "look_at_player")
        self.assertEqual(manager.state, "idle")


if __name__ == "__main__":
    unittest.main(verbosity=2)
