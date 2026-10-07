"""无需 Camera、Microphone 或 Unity 的 Agent 独立测试入口。"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.speech_recognition import SpeechResult
from events.interaction_event import InteractionEvent
from npc.action_schema import parse_agent_output, parse_agent_output_detailed
from npc.agent import NpcAgent
from npc.conversation_context import ConversationContext
from npc.llm_provider import OpenAICompatibleProvider


def make_event(
    text,
    person_detected=True,
    left_hand="down",
    right_hand="down",
    eye_contact="true",
    directed_speech="true",
    language="zh",
    relationship_level="stranger",
):
    state = SimpleNamespace(
        person_detected=person_detected,
        face_detected=person_detected,
        horizontal_position="center" if person_detected else "unknown",
        distance="medium" if person_detected else "unknown",
        left_hand=left_hand,
        right_hand=right_hand,
        eye_contact=eye_contact,
        looking_at_npc="true" if eye_contact == "true" else "unknown",
        directed_speech=directed_speech,
    )
    return InteractionEvent.speech(
        SpeechResult(text, language),
        state,
        relationship_level=relationship_level,
    )


def run_self_test():
    invalid = parse_agent_output(
        'Result: ```json\n{"reply":"你好呀","action":"attack","emotion":"angry"}\n```'
    )
    assert invalid.action == "none" and invalid.emotion == "neutral"
    invalid_details = parse_agent_output_detailed(
        '{"reply":"你好呀","action":"attack","emotion":"happy"}'
    )
    assert invalid_details.action_error_count == 1

    class FakeProvider:
        def generate(self, messages):
            return '{"reply":"左手。","action":"wave","emotion":"happy"}'

    agent = NpcAgent(provider=FakeProvider(), llm_enabled=True)
    event = make_event("我举的是哪只手？", left_hand="raised")
    result = agent._make_decision(event)
    assert result.decision.action == "wave"

    class TimeoutProvider:
        def generate(self, messages):
            raise TimeoutError("simulated timeout")

    fallback = NpcAgent(provider=TimeoutProvider(), llm_enabled=True)
    result = fallback._make_decision(make_event("你好"))
    assert result.decision.source == "rule"

    class ContradictingProvider:
        def generate(self, messages):
            return '{"reply":"我看到你了。","action":"look_at_player","emotion":"happy"}'

    guarded = NpcAgent(provider=ContradictingProvider(), llm_enabled=True)
    result = guarded._make_decision(
        make_event(
            "你能看到我吗",
            person_detected=False,
            eye_contact="unknown",
            directed_speech="unknown",
        )
    )
    assert result.decision.source == "rule"
    assert result.decision.reply == "我现在没有检测到你。"
    assert result.fact_violation_count == 1

    class EyeContactGuessProvider:
        def generate(self, messages):
            return '{"reply":"你正在看我的眼睛。","action":"nod","emotion":"happy"}'

    eye_guarded = NpcAgent(provider=EyeContactGuessProvider(), llm_enabled=True)
    result = eye_guarded._make_decision(
        make_event("我现在是不是在看你？", eye_contact="unknown")
    )
    assert result.decision.source == "rule"

    class HandGuessProvider:
        def generate(self, messages):
            return '{"reply":"你举的是左手。","action":"wave","emotion":"happy"}'

    hand_guarded = NpcAgent(provider=HandGuessProvider(), llm_enabled=True)
    result = hand_guarded._make_decision(
        make_event("我举的是哪只手？", left_hand="unknown")
    )
    assert result.decision.source == "rule"

    disabled = NpcAgent(llm_enabled=False)
    result = disabled._make_decision(make_event("你好"))
    assert result.decision.source == "rule"

    memory = ConversationContext(max_turns=6)
    for index in range(7):
        memory.add_turn(f"u{index}", f"a{index}")
    assert len(memory.messages()) == 6
    assert memory.messages()[0]["content"] == "u4"

    class CountingProvider:
        def __init__(self):
            self.calls = 0
            self.messages = None

        def generate(self, messages):
            self.calls += 1
            self.messages = messages
            return '{"reply":"你好呀。","action":"nod","emotion":"curious"}'

    provider = CountingProvider()
    asynchronous = NpcAgent(provider=provider, llm_enabled=True)
    asynchronous.start()
    assert asynchronous.submit(make_event("你好"))
    phases = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and "completed" not in phases:
        lifecycle = asynchronous.poll_event()
        if lifecycle is None:
            time.sleep(0.01)
            continue
        phases.append(lifecycle.phase)
    asynchronous.shutdown()
    assert provider.calls == 1
    assert phases == ["started", "completed"]
    assert provider.messages[-1] == {"role": "user", "content": "你好"}
    assert any(
        "AUTHORITATIVE PERCEPTION CONTEXT" in item["content"]
        for item in provider.messages
        if item["role"] == "system"
    )
    assert any(
        "response_tone=polite" in item["content"]
        for item in provider.messages
        if item["role"] == "system"
    )

    blocked_provider = CountingProvider()
    blocked = NpcAgent(provider=blocked_provider, llm_enabled=True)
    assert not blocked.submit(make_event("你好", directed_speech="false"))
    assert blocked_provider.calls == 0

    class FakeHttpResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "provider-ok"}}]}
            ).encode("utf-8")

    captured_request = {}

    def fake_urlopen(request, timeout):
        captured_request["url"] = request.full_url
        captured_request["body"] = json.loads(request.data.decode("utf-8"))
        captured_request["timeout"] = timeout
        return FakeHttpResponse()

    http_provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:9999/v1",
        model="test-model",
        api_key_env="",
        timeout=1.5,
    )
    with patch("npc.llm_provider.urllib.request.urlopen", fake_urlopen):
        assert http_provider.generate([{"role": "user", "content": "hi"}]) == "provider-ok"
    assert captured_request["url"].endswith("/v1/chat/completions")
    assert captured_request["body"]["model"] == "test-model"
    assert captured_request["body"]["stream"] is False

    print(
        "Agent parser, whitelist, perception guard, timeout, fallback, "
        "memory, async single-call, directed gate, provider request: OK"
    )


def interactive():
    print("Agent 独立测试（当前配置无LLM时自动使用规则Fallback）。")
    text = input("Speech Text: ").strip()
    if not text:
        print("输入为空。")
        return
    visible = input("Person visible? [Y/n]: ").strip().lower() != "n"
    eye = input("Eye contact [true/false/unknown]: ").strip() or "unknown"
    left = input("Left hand [raised/down/unknown]: ").strip() or "unknown"
    event = make_event(
        text,
        person_detected=visible,
        left_hand=left,
        eye_contact=eye,
        directed_speech="true" if eye == "true" else "unknown",
    )
    agent = NpcAgent()
    result = agent._make_decision(event)
    print(result.decision.to_dict() if result.decision else "No response")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    run_self_test() if args.self_test else interactive()
