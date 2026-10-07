"""异步 Multimodal NPC Agent：LLM 决策、验证与规则降级。"""

import queue
import threading
import time
from dataclasses import dataclass

from audio.audio_config import RESPOND_TO_UNKNOWN_DIRECTED_SPEECH
from behavior.relationship_behavior import build_relationship_behavior_profile

from .action_schema import (
    AgentDecision,
    FactViolationError,
    enforce_perception_constraints,
    parse_agent_output_detailed,
)
from .agent_config import (
    AGENT_QUEUE_SIZE,
    DEBUG_AGENT,
    LLM_ENABLED,
    LLM_PROVIDER,
)
from .conversation_context import ConversationContext
from .llm_provider import OpenAICompatibleProvider
from .personality import AGENT_SAFETY_INSTRUCTIONS, get_default_personality
from .response_engine import RuleBasedResponseEngine
from .world_context import build_world_context


@dataclass(frozen=True)
class AgentResult:
    speech_event: object
    decision: AgentDecision | None
    latency_ms: int
    fallback_reason: str | None = None
    action_error_count: int = 0
    fact_violation_count: int = 0


@dataclass(frozen=True)
class AgentLifecycle:
    phase: str
    speech_event: object
    result: AgentResult | None = None


class NpcAgent:
    """每个 Speech Event 只入队一次；工作线程不会阻塞视觉主循环。"""

    def __init__(self, provider=None, llm_enabled=None):
        self.llm_enabled = LLM_ENABLED if llm_enabled is None else llm_enabled
        self.provider = provider
        self.personality = get_default_personality()
        self.conversation = ConversationContext()
        self.rule_engine = RuleBasedResponseEngine()
        self._jobs = queue.Queue(maxsize=AGENT_QUEUE_SIZE)
        self._events = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = None

    def start(self):
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="NpcAgent",
            daemon=True,
        )
        self._worker.start()

    def submit(self, speech_event):
        """只接受明确 Speech Event，并在这里执行 Directed Speech Gate。"""
        if speech_event is None or speech_event.event_type != "speech":
            return False
        directed = speech_event.context.directed_speech
        if directed == "false":
            return False
        if directed == "unknown" and not RESPOND_TO_UNKNOWN_DIRECTED_SPEECH:
            return False

        try:
            self._jobs.put_nowait(speech_event)
            return True
        except queue.Full:
            self.conversation.add_user_turn(speech_event.text)
            result = self._fallback(speech_event, "Agent queue is full")
            self._events.put(AgentLifecycle("completed", speech_event, result))
            return True

    def poll_event(self):
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                speech_event = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if speech_event is None:
                self._jobs.task_done()
                break

            self._events.put(AgentLifecycle("started", speech_event))
            result = self._make_decision(speech_event)
            self._events.put(AgentLifecycle("completed", speech_event, result))
            self._jobs.task_done()

    def _make_decision(self, speech_event):
        started = time.monotonic()
        action_error_count = 0
        self.conversation.add_user_turn(speech_event.text)
        if not self.llm_enabled:
            return self._fallback(speech_event, "LLM disabled", started)

        try:
            provider = self.provider or self._create_provider()
            world_context = build_world_context(speech_event)
            relationship_profile = build_relationship_behavior_profile(
                speech_event.context.relationship_level,
                interaction_context=speech_event.context,
            )
            messages = [
                {"role": "system", "content": self.personality.to_context()},
                {"role": "system", "content": AGENT_SAFETY_INSTRUCTIONS},
                {
                    "role": "system",
                    "content": (
                        "AUTHORITATIVE PERCEPTION CONTEXT:\n"
                        + world_context
                        + "\nNever contradict these facts."
                    ),
                },
                {
                    "role": "system",
                    "content": (
                        "RELATIONSHIP BEHAVIOR CONTEXT:\n"
                        f"relationship_level={relationship_profile.relationship_level}\n"
                        f"response_tone={relationship_profile.response_tone}\n"
                        "Use the response tone lightly; never override perception facts."
                    ),
                },
            ]
            # 当前 user Turn 已在短期 Context 中；玩家文本绝不拼入 System Prompt。
            messages.extend(
                {
                    "role": turn["role"],
                    "content": turn["content"],
                }
                for turn in self.conversation.get_recent_context()
            )
            raw_output = provider.generate(messages)
            parsed = parse_agent_output_detailed(raw_output)
            action_error_count = parsed.action_error_count
            decision = parsed.decision
            decision = enforce_perception_constraints(decision, speech_event)
            latency = int((time.monotonic() - started) * 1000)
            self.conversation.add_assistant_turn(decision.reply)
            self._debug_log(speech_event, world_context, decision, latency)
            return AgentResult(
                speech_event,
                decision,
                latency,
                action_error_count=parsed.action_error_count,
            )
        except FactViolationError as error:
            return self._fallback(
                speech_event,
                str(error),
                started,
                action_error_count=action_error_count,
                fact_violation_count=1,
            )
        except Exception as error:
            return self._fallback(speech_event, str(error), started)

    def _fallback(
        self,
        speech_event,
        reason,
        started=None,
        action_error_count=0,
        fact_violation_count=0,
    ):
        started = time.monotonic() if started is None else started
        response_event = self.rule_engine.create_response(speech_event)
        decision = None
        if response_event is not None:
            decision = AgentDecision(
                reply=response_event.text,
                action="none",
                emotion="neutral",
                source="rule",
            )
            self.conversation.add_assistant_turn(decision.reply)
        latency = int((time.monotonic() - started) * 1000)
        print(f"Agent fallback: {reason}")
        return AgentResult(
            speech_event,
            decision,
            latency,
            reason,
            action_error_count,
            fact_violation_count,
        )

    def _create_provider(self):
        if LLM_PROVIDER != "openai_compatible":
            raise ValueError(f"不支持的 LLM_PROVIDER：{LLM_PROVIDER}")
        self.provider = OpenAICompatibleProvider()
        return self.provider

    def _debug_log(self, speech_event, world_context, decision, latency):
        if not DEBUG_AGENT:
            return
        print(f"Speech:\n{speech_event.text}")
        print(f"World Context:\n{world_context}")
        print("Agent:\nLLM")
        print(
            "Decision:\n"
            f"reply={decision.reply}\n"
            f"action={decision.action}\n"
            f"emotion={decision.emotion}"
        )
        print(f"Latency:\n{latency} ms")

    def shutdown(self):
        self._stop_event.set()
        if self._worker is not None:
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=2.0)
        self.conversation.clear()
