"""不使用摄像头、麦克风和 TTS，重新运行历史 Speech Event。"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.interaction_logger import InteractionLogger
from events.interaction_event import InteractionEvent
from npc.agent import NpcAgent
from npc.world_context import build_world_context


DEFAULT_CASES = Path(__file__).resolve().parent / "test_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "replay_interactions.jsonl"


class FixedOutputProvider:
    def __init__(self, output):
        self.output = output

    def generate(self, messages):
        return self.output


def load_inputs(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                logged = json.loads(line)
                records.append(
                    {
                        "id": f"history-{len(records) + 1}",
                        "speech_text": logged["speech_text"],
                        "language": logged.get("speech_language", "zh"),
                        "world_state": logged.get("speech_context", {}),
                    }
                )
        return records
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_speech_event(case):
    state_values = {
        "person_detected": True,
        "face_detected": True,
        "horizontal_position": "center",
        "distance": "medium",
        "left_hand": "down",
        "right_hand": "down",
        "eye_contact": "unknown",
        "looking_at_npc": "unknown",
        "directed_speech": "true",
    }
    state_values.update(case.get("world_state", {}))
    speech_result = SimpleNamespace(
        text=case["speech_text"],
        language=case.get("language", "zh"),
        confidence=None,
    )
    return InteractionEvent.speech(
        speech_result,
        SimpleNamespace(**state_values),
    )


def replay(path=DEFAULT_CASES, output=DEFAULT_OUTPUT):
    logger = InteractionLogger(output)
    results = []
    for case in load_inputs(path):
        speech_event = build_speech_event(case)
        provider = None
        llm_enabled = case.get("llm_enabled")
        if "mock_output" in case:
            provider = FixedOutputProvider(case["mock_output"])
            llm_enabled = True

        agent = NpcAgent(provider=provider, llm_enabled=llm_enabled)
        started = time.monotonic()
        result = agent._make_decision(speech_event)
        elapsed = int((time.monotonic() - started) * 1000)
        world_context = build_world_context(speech_event)
        record = logger.write(
            speech_event,
            world_context,
            result,
            stt_latency=0,
            tts_latency=0,
            total_latency=elapsed,
        )
        results.append(record)
        decision = result.decision
        print(
            f"[{case.get('id', '?')}] "
            f"source={decision.source if decision else 'none'} "
            f"action={decision.action if decision else 'none'} "
            f"reply={decision.reply if decision else ''}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线重放 Speech Event")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    replay(args.input, args.output)
