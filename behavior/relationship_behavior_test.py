"""Relationship Behavior 纯策略和最小集成测试。"""

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.behavior_manager import BehaviorManager
from behavior.relationship_behavior import build_relationship_behavior_profile
from events.world_event import WorldEvent


class RelationshipBehaviorProfileTests(unittest.TestCase):
    def test_stranger_profile(self):
        profile = build_relationship_behavior_profile("stranger")
        self.assertFalse(profile.allow_proactive_greeting)
        self.assertFalse(profile.allow_proactive_wave)
        self.assertEqual(profile.preferred_social_distance, "normal")
        self.assertEqual(profile.response_tone, "polite")
        self.assertEqual(profile.attention_priority, "normal")

    def test_acquaintance_profile(self):
        profile = build_relationship_behavior_profile("acquaintance")
        self.assertTrue(profile.allow_proactive_greeting)
        self.assertTrue(profile.allow_proactive_wave)
        self.assertEqual(profile.preferred_social_distance, "familiar")
        self.assertEqual(profile.response_tone, "friendly")
        self.assertEqual(profile.attention_priority, "slightly_high")

    def test_friend_profile(self):
        profile = build_relationship_behavior_profile("friend")
        self.assertTrue(profile.allow_proactive_greeting)
        self.assertTrue(profile.allow_proactive_wave)
        self.assertEqual(profile.preferred_social_distance, "close")
        self.assertEqual(profile.response_tone, "warm")
        self.assertEqual(profile.attention_priority, "high")

    def test_invalid_level_falls_back_to_stranger(self):
        profile = build_relationship_behavior_profile("unknown-level")
        self.assertEqual(profile.relationship_level, "stranger")
        self.assertFalse(profile.allow_proactive_greeting)
        self.assertFalse(profile.allow_proactive_wave)

    def test_profile_does_not_modify_memory_data(self):
        memory_data = {
            "relationship_level": "friend",
            "times_seen": 8,
            "times_spoken": 12,
        }
        original = copy.deepcopy(memory_data)
        build_relationship_behavior_profile(
            memory_data["relationship_level"],
            interaction_context=memory_data,
        )
        self.assertEqual(memory_data, original)


class RelationshipBehaviorIntegrationTests(unittest.TestCase):
    def test_stranger_blocks_proactive_greeting_and_wave(self):
        manager = BehaviorManager()
        greeting = WorldEvent.create(
            "player_enter",
            {"relationship_level": "stranger"},
        )
        wave = WorldEvent.create(
            "hand_wave",
            {"relationship_level": "stranger"},
        )
        self.assertIsNone(manager.handle_events([greeting], now=0.0))
        self.assertIsNone(manager.handle_events([wave], now=0.1))

    def test_acquaintance_allows_proactive_greeting_and_wave(self):
        greeting_manager = BehaviorManager()
        greeting = WorldEvent.create(
            "player_enter",
            {"relationship_level": "acquaintance"},
        )
        greeting_decision = greeting_manager.handle_events(
            [greeting],
            now=0.0,
        )
        self.assertEqual(greeting_decision.action, "wave")
        self.assertTrue(greeting_decision.reply)

        wave_manager = BehaviorManager()
        wave = WorldEvent.create(
            "hand_wave",
            {"relationship_level": "acquaintance"},
        )
        wave_decision = wave_manager.handle_events([wave], now=0.0)
        self.assertEqual(wave_decision.action, "wave")

    def test_friend_allows_proactive_greeting_and_wave(self):
        for event_type in ("player_enter", "hand_wave"):
            manager = BehaviorManager()
            event = WorldEvent.create(
                event_type,
                {"relationship_level": "friend"},
            )
            self.assertIsNotNone(manager.handle_events([event], now=0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
