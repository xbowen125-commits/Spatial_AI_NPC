"""将一次完整交互以 JSONL 形式追加到本地日志。"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "interactions.jsonl"
DEFAULT_BEHAVIOR_LOG_PATH = PROJECT_ROOT / "logs" / "behaviors.jsonl"
DEFAULT_EMOTION_LOG_PATH = PROJECT_ROOT / "logs" / "emotions.jsonl"


class InteractionLogger:
    """只记录文本、语义状态和耗时，不接收音频、图像或密钥。"""

    def __init__(self, path=DEFAULT_LOG_PATH):
        self.path = Path(path)

    def write(
        self,
        speech_event,
        world_context,
        agent_result,
        stt_latency=0,
        tts_latency=0,
        total_latency=None,
    ):
        decision = agent_result.decision
        if decision is None:
            return None

        stt_latency = _non_negative_int(stt_latency)
        agent_latency = _non_negative_int(agent_result.latency_ms)
        tts_latency = _non_negative_int(tts_latency)
        if total_latency is None:
            total_latency = stt_latency + agent_latency + tts_latency

        context = speech_event.context
        if is_dataclass(context):
            context = asdict(context)
        elif not isinstance(context, dict):
            context = dict(vars(context))

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speech_event_timestamp": int(speech_event.timestamp),
            "speech_text": str(speech_event.text),
            "speech_language": str(speech_event.language),
            "speech_context": context,
            "world_context": str(world_context),
            "agent_source": decision.source,
            "reply": decision.reply,
            "action": decision.action,
            "emotion": decision.emotion,
            "fallback_reason": agent_result.fallback_reason,
            "latency": {
                "stt_latency": stt_latency,
                "agent_latency": agent_latency,
                "tts_latency": tts_latency,
                "total_latency": _non_negative_int(total_latency),
            },
            "action_error_count": int(agent_result.action_error_count),
            "fact_violation_count": int(agent_result.fact_violation_count),
        }

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as error:
            # 评估记录失败不能中断视觉、Agent、TTS 或 Unity 主链路。
            print(f"Evaluation log failed: {error}")
            return None
        return record


class BehaviorLogger:
    """记录主动事件的执行、排队、冷却或忽略结果。"""

    def __init__(self, path=DEFAULT_BEHAVIOR_LOG_PATH):
        self.path = Path(path)

    def write(
        self,
        world_event,
        behavior_decision,
        cooldown_status,
        cooldown_remaining=0.0,
        behavior_state="idle",
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": world_event.event_type,
            "event_priority": world_event.priority,
            "event_timestamp": world_event.timestamp,
            "event_context": world_event.context,
            "behavior_decision": (
                behavior_decision.to_dict()
                if behavior_decision is not None
                else None
            ),
            "action": (
                behavior_decision.action
                if behavior_decision is not None
                else "none"
            ),
            "reply": (
                behavior_decision.reply
                if behavior_decision is not None
                else ""
            ),
            "cooldown_status": str(cooldown_status),
            "cooldown_remaining": round(
                max(0.0, float(cooldown_remaining)),
                3,
            ),
            "npc_behavior_state": str(behavior_state),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as error:
            print(f"Behavior log failed: {error}")
            return None
        return record


class EmotionLogger:
    """记录情绪状态转换，不接收或保存Speech Text。"""

    def __init__(self, path=DEFAULT_EMOTION_LOG_PATH):
        self.path = Path(path)

    def write(
        self,
        trigger_event,
        previous_state,
        new_state,
        relationship_level,
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_event": str(trigger_event),
            "previous_emotion": previous_state.emotion,
            "new_emotion": new_state.emotion,
            "previous_intensity": round(previous_state.intensity, 4),
            "new_intensity": round(new_state.intensity, 4),
            "relationship_level": str(relationship_level),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as error:
            print(f"Emotion log failed: {error}")
            return None
        return record


def read_jsonl(path=DEFAULT_LOG_PATH):
    """逐行读取合法 JSON 对象；损坏行留给 metrics 报告。"""
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def _non_negative_int(value):
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
