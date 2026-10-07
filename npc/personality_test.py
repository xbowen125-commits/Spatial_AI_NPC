"""结构化 Personality 与 Agent 组合上下文测试。"""

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
from behavior.relationship_behavior import build_relationship_behavior_profile
from events.interaction_event import InteractionEvent
from memory.memory_policy import MemoryCandidate
from memory.memory_store import MemoryStore
from npc.agent import NpcAgent
from npc.conversation_context import ConversationContext
from npc.personality import AIRI, SpeakingStyle, get_default_personality


def make_event(text, relationship_level):
    state = SimpleNamespace(
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
        state,
        relationship_level=relationship_level,
    )


class FakeProvider:
    def __init__(self):
        self.requests = []

    def generate(self, messages):
        self.requests.append(list(messages))
        return json.dumps(
            {
                "reply": "好的。",
                "action": "none",
                "emotion": "neutral",
            },
            ensure_ascii=False,
        )


class PersonalityTests(unittest.TestCase):
    def test_default_profile_identity_and_name(self):
        profile = get_default_personality()
        self.assertIs(profile, AIRI)
        self.assertEqual(profile.name, "Airi")
        self.assertEqual(
            profile.identity,
            "A virtual companion living in a spatial world.",
        )

    def test_traits_and_speaking_style(self):
        profile = get_default_personality()
        self.assertEqual(
            profile.traits,
            ("calm", "curious", "observant", "concise"),
        )
        self.assertIsInstance(profile.speaking_style, SpeakingStyle)
        self.assertEqual(profile.speaking_style.verbosity, "concise")
        self.assertEqual(profile.speaking_style.warmth, "medium")
        self.assertEqual(profile.speaking_style.humor, "light")

    def test_context_is_stable_and_structured(self):
        profile = get_default_personality()
        first = profile.to_context()
        second = profile.to_context()
        self.assertEqual(first, second)
        for value in ("Airi", "calm", "curious", "concise"):
            self.assertIn(value, first)
        self.assertTrue(first.startswith("PERSONALITY CONTEXT:"))

    def test_personality_does_not_modify_relationship_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "player_profile.json"
            store = MemoryStore(path)
            store.remember(MemoryCandidate("interaction", "seen", timestamp=1))
            before = path.read_bytes()
            get_default_personality().to_context()
            self.assertEqual(path.read_bytes(), before)

    def test_personality_does_not_modify_conversation_context(self):
        conversation = ConversationContext(session_id="test-session")
        conversation.add_user_turn("你好", timestamp=1)
        before = copy.deepcopy(conversation.get_recent_context())
        get_default_personality().to_context()
        self.assertEqual(conversation.get_recent_context(), before)

    def test_relationship_changes_do_not_change_personality(self):
        profile = get_default_personality()
        before = profile.to_dict()
        stranger = build_relationship_behavior_profile("stranger")
        friend = build_relationship_behavior_profile("friend")
        self.assertNotEqual(stranger.response_tone, friend.response_tone)
        self.assertEqual(profile.to_dict(), before)
        self.assertEqual(profile.to_context(), get_default_personality().to_context())

    def test_agent_receives_separate_context_sources(self):
        provider = FakeProvider()
        agent = NpcAgent(provider=provider, llm_enabled=True)
        agent._make_decision(make_event("你叫什么？", "friend"))
        agent._make_decision(make_event("那你为什么在这里？", "friend"))
        messages = provider.requests[-1]

        personality = [
            item for item in messages
            if item["role"] == "system"
            and item["content"].startswith("PERSONALITY CONTEXT:")
        ]
        relationship = [
            item for item in messages
            if item["role"] == "system"
            and item["content"].startswith("RELATIONSHIP BEHAVIOR CONTEXT:")
        ]
        perception = [
            item for item in messages
            if item["role"] == "system"
            and item["content"].startswith("AUTHORITATIVE PERCEPTION CONTEXT:")
        ]
        self.assertEqual(len(personality), 1)
        self.assertEqual(len(relationship), 1)
        self.assertEqual(len(perception), 1)
        self.assertIn("name=Airi", personality[0]["content"])
        self.assertIn("traits=calm, curious", personality[0]["content"])
        self.assertIn("relationship_level=friend", relationship[0]["content"])
        self.assertIn("response_tone=warm", relationship[0]["content"])
        self.assertIn(
            {"role": "user", "content": "你叫什么？"},
            messages,
        )
        self.assertIn(
            {"role": "assistant", "content": "好的。"},
            messages,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
