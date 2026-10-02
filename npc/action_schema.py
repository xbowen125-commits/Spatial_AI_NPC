"""Agent 结构化决策与严格白名单验证。"""

import json
from dataclasses import asdict, dataclass, replace

from .agent_config import MAX_AGENT_REPLY_CHARS


ALLOWED_ACTIONS = frozenset({"none", "wave", "nod", "look_at_player"})
ALLOWED_EMOTIONS = frozenset({"neutral", "happy", "curious", "surprised"})


class DecisionValidationError(ValueError):
    pass


class FactViolationError(DecisionValidationError):
    """模型回复与权威感知事实冲突。"""


@dataclass(frozen=True)
class AgentDecision:
    reply: str
    action: str
    emotion: str
    source: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ParsedAgentOutput:
    decision: AgentDecision
    action_error_count: int = 0


def parse_agent_output_detailed(raw_text):
    """从纯JSON、Markdown代码块或带额外文字的输出中提取首个JSON对象。"""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise DecisionValidationError("模型返回为空。")

    decoder = json.JSONDecoder()
    data = None
    text = raw_text.strip()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            data = candidate
            break

    if data is None:
        raise DecisionValidationError("未找到合法JSON对象。")

    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise DecisionValidationError("reply 缺失或为空。")
    reply = reply.strip()[:MAX_AGENT_REPLY_CHARS]

    action = str(data.get("action", "none")).strip().lower()
    action_error_count = 0
    if action not in ALLOWED_ACTIONS:
        action = "none"
        action_error_count = 1

    emotion = str(data.get("emotion", "neutral")).strip().lower()
    if emotion not in ALLOWED_EMOTIONS:
        emotion = "neutral"

    return ParsedAgentOutput(
        AgentDecision(reply, action, emotion, "llm"),
        action_error_count,
    )


def parse_agent_output(raw_text):
    """保持原有接口：只返回验证后的 AgentDecision。"""
    return parse_agent_output_detailed(raw_text).decision


def enforce_perception_constraints(decision, speech_event):
    """对常见感知事实声明做保守检查；冲突时交给规则引擎降级。"""
    context = speech_event.context
    reply = decision.reply.lower()

    if not context.person_detected and _contains_any(
        reply,
        (
            "我看到你",
            "我看见你",
            "我能看到你",
            "我能看见你",
            "i see you",
            "i can see you",
        ),
    ):
        raise FactViolationError("回复与 person_detected=false 冲突。")

    if context.eye_contact != "true" and _contains_any(
        reply,
        (
            "正在看着你",
            "和你对视",
            "你在看我",
            "你正看着我",
            "你正在看着我",
            "你在看我的眼睛",
            "你正在看我的眼睛",
            "你和我对视",
            "making eye contact",
            "looking at you",
            "you are looking at me",
            "you're looking at me",
            "you are making eye contact",
            "you're making eye contact",
        ),
    ):
        raise FactViolationError("回复与 eye_contact 感知事实冲突。")

    if context.left_hand == "unknown" and _contains_any(
        reply,
        ("左手", "left hand"),
    ):
        raise FactViolationError("左手状态未知，不能猜测。")
    if context.right_hand == "unknown" and _contains_any(
        reply,
        ("右手", "right hand"),
    ):
        raise FactViolationError("右手状态未知，不能猜测。")

    if not context.person_detected and decision.action == "look_at_player":
        return replace(decision, action="none")
    return decision


def _contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)
