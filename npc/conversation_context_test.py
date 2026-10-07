"""无需真实 LLM、摄像头、麦克风或磁盘日志的短期上下文测试。"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.speech_recognition import SpeechResult
from events.interaction_event import InteractionEvent
from memory.memory_policy import MemoryCandidate
from memory.memory_store import MemoryStore
from npc.agent import NpcAgent
from npc.conversation_context import MAX_TURNS, ConversationContext


def make_event(text, relationship_level="friend"):
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
    def __init__(self, replies=None):
        self.replies = list(replies or ["好的。"])
        self.requests = []

    def generate(self, messages):
        self.requests.append(list(messages))
        reply = self.replies.pop(0)
        return json.dumps(
            {
                "reply": reply,
                "action": "none",
                "emotion": "neutral",
            },
            ensure_ascii=False,
        )


class ConversationContextTests(unittest.TestCase):
    def test_empty_context_has_session_id(self):
        context = ConversationContext(session_id="test-session")
        self.assertEqual(context.session_id, "test-session")
        self.assertEqual(context.get_recent_context(), [])
        self.assertEqual(len(context), 0)

    def test_add_user_and_assistant_turns_in_order(self):
        context = ConversationContext()
        user = context.add_user_turn("你好", timestamp=100)
        assistant = context.add_assistant_turn("你好呀", timestamp=200)
        self.assertEqual(user.role, "user")
        self.assertEqual(assistant.role, "assistant")
        self.assertEqual(
            context.get_recent_context(),
            [
                {"role": "user", "content": "你好", "timestamp": 100},
                {
                    "role": "assistant",
                    "content": "你好呀",
                    "timestamp": 200,
                },
            ],
        )

    def test_recent_context_is_a_copy(self):
        context = ConversationContext()
        context.add_user_turn("原始内容", timestamp=1)
        returned = context.get_recent_context()
        returned[0]["content"] = "外部修改"
        self.assertEqual(
            context.get_recent_context()[0]["content"],
            "原始内容",
        )

    def test_max_turns_discards_oldest_turns(self):
        context = ConversationContext(max_turns=3)
        for index in range(5):
            context.add_user_turn(f"turn-{index}", timestamp=index)
        self.assertEqual(len(context), 3)
        self.assertEqual(
            [turn["content"] for turn in context.get_recent_context()],
            ["turn-2", "turn-3", "turn-4"],
        )
        self.assertEqual(MAX_TURNS, 10)

    def test_clear_removes_turns_but_keeps_session(self):
        context = ConversationContext(session_id="same-session")
        context.add_user_turn("问题")
        context.add_assistant_turn("回答")
        context.clear()
        self.assertEqual(context.get_recent_context(), [])
        self.assertEqual(context.session_id, "same-session")

    def test_context_does_not_modify_relationship_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "player_profile.json"
            store = MemoryStore(memory_path)
            store.remember(MemoryCandidate("interaction", "seen", timestamp=1))
            before = memory_path.read_bytes()

            context = ConversationContext()
            context.add_user_turn("这是短期对话")
            context.add_assistant_turn("不会进入长期记忆")

            self.assertEqual(memory_path.read_bytes(), before)
            profile_text = memory_path.read_text(encoding="utf-8")
            self.assertNotIn("这是短期对话", profile_text)
            self.assertNotIn("不会进入长期记忆", profile_text)

    def test_context_creates_no_disk_files(self):
        with tempfile.TemporaryDirectory() as directory:
            before = set(os.listdir(directory))
            context = ConversationContext()
            context.add_user_turn("只在内存")
            context.add_assistant_turn("确认")
            self.assertEqual(set(os.listdir(directory)), before)

    def test_agent_reads_context_and_relationship_tone(self):
        provider = FakeProvider(["我叫Airi。", "我在这里陪你。"])
        agent = NpcAgent(provider=provider, llm_enabled=True)

        first = agent._make_decision(make_event("你叫什么？"))
        second = agent._make_decision(make_event("那你为什么在这里？"))

        self.assertEqual(first.decision.reply, "我叫Airi。")
        self.assertEqual(second.decision.reply, "我在这里陪你。")
        second_request = provider.requests[1]
        self.assertIn(
            {"role": "user", "content": "你叫什么？"},
            second_request,
        )
        self.assertIn(
            {"role": "assistant", "content": "我叫Airi。"},
            second_request,
        )
        self.assertEqual(
            second_request[-1],
            {"role": "user", "content": "那你为什么在这里？"},
        )
        self.assertTrue(
            any(
                "relationship_level=friend" in item["content"]
                and "response_tone=warm" in item["content"]
                for item in second_request
                if item["role"] == "system"
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
