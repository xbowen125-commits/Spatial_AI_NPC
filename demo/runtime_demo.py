"""使用 Fake PlayerState 验证 Runtime 完整决策链，无硬件或外部 API。"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion.emotion_engine import EmotionEngine
from events.world_event import WorldEvent
from runtime.npc_runtime import NPCManager


@dataclass(frozen=True)
class FakePlayerState:
    """只提供此 Demo 所需的感知字段。"""

    person_detected: bool = False
    eye_contact: str = "false"
    left_hand: str = "down"
    right_hand: str = "down"
    is_speaking: bool = False


def run_demo():
    """在同一 Runtime 中顺序运行三个场景，固定时间保证可重复。"""
    runtime = NPCManager(emotion_engine=EmotionEngine(now=0.0))
    scenarios = (
        ("player_enter", "player_enter", FakePlayerState(True, "true"),
         "stranger", 0.0, "look_at_player", None),
        # 项目中的挥手事件正式名称为 hand_wave，复用既有 Schema。
        ("player_wave + friend", "hand_wave",
         FakePlayerState(True, "true", left_hand="raised"),
         "friend", 1.0, "look_at_player", "wave"),
        ("player_leave", "player_leave", FakePlayerState(),
         "friend", 2.0, "idle", None),
    )
    outputs = []
    for index, (name, event_type, state, relationship, now,
                expected_attention, expected_interaction) in enumerate(scenarios, 1):
        event = WorldEvent.create(event_type, {}, timestamp=index)
        result = runtime.tick(
            player_state=state,
            world_events=(event,),
            relationship_state=relationship,
            conversation_state=False,
            behavior_state="idle",
            now=now,
        )
        decision = result.autonomous_decision
        if result.error or decision is None:
            raise RuntimeError(f"{name}: Runtime failed: {result.error}")
        if (decision.attention, decision.interaction) != (
            expected_attention, expected_interaction
        ):
            raise AssertionError(f"{name}: unexpected decision {decision}")
        expected_action = expected_interaction or expected_attention
        if expected_action not in result.dispatch_actions:
            raise AssertionError(f"{name}: missing dispatch {expected_action}")
        row = {
            "scenario": name,
            "events_processed": list(result.events_processed),
            "attention": decision.attention,
            "interaction": decision.interaction,
            "selected_action": decision.arbitration.action,
            "dispatch_actions": list(result.dispatch_actions),
            "reasons": list(decision.reasons),
            "passed": True,
        }
        outputs.append(row)
        print(json.dumps(row, ensure_ascii=False))
    print("Runtime demo: 3/3 scenarios passed.")
    return outputs


if __name__ == "__main__":
    run_demo()
