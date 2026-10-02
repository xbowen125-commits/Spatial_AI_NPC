"""Python → Unity 的轻量 Interaction Event。"""

import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SpeechContext:
    person_detected: bool
    face_detected: bool
    horizontal_position: str
    eye_contact: str
    looking_at_npc: str
    distance: str
    left_hand: str
    right_hand: str
    directed_speech: str
    relationship_level: str = "stranger"
    npc_emotion: str = "neutral"
    npc_emotion_intensity: float = 0.0


@dataclass(frozen=True)
class InteractionEvent:
    message_type: str
    event_type: str
    timestamp: int
    text: str
    language: str
    confidence: float | None
    context: SpeechContext

    @classmethod
    def speech(
        cls,
        speech_result,
        player_state,
        relationship_level="stranger",
        npc_emotion="neutral",
        npc_emotion_intensity=0.0,
    ):
        return cls(
            message_type="interaction_event",
            event_type="speech",
            timestamp=int(time.time() * 1000),
            text=speech_result.text,
            language=speech_result.language,
            confidence=speech_result.confidence,
            context=SpeechContext(
                person_detected=player_state.person_detected,
                face_detected=player_state.face_detected,
                horizontal_position=player_state.horizontal_position,
                eye_contact=player_state.eye_contact,
                looking_at_npc=player_state.looking_at_npc,
                distance=player_state.distance,
                left_hand=player_state.left_hand,
                right_hand=player_state.right_hand,
                directed_speech=player_state.directed_speech,
                relationship_level=str(relationship_level),
                npc_emotion=str(npc_emotion),
                npc_emotion_intensity=float(npc_emotion_intensity),
            ),
        )

    def to_dict(self):
        data = asdict(self)
        # relationship_level 只供本地 Agent World Context 使用，
        # 不扩展既有 Python → Unity Speech Event 通信结构。
        data["context"].pop("relationship_level", None)
        data["context"].pop("npc_emotion", None)
        data["context"].pop("npc_emotion_intensity", None)
        return data


@dataclass(frozen=True)
class ResponseEvent:
    """Agent 或规则 Fallback 产生的 NPC 文字回复。"""

    message_type: str
    event_type: str
    timestamp: int
    text: str
    language: str
    source_event: str
    subtitle_duration: float

    @classmethod
    def create(cls, text, language, source_event="speech"):
        from audio.audio_config import SUBTITLE_DURATION

        return cls(
            message_type="interaction_event",
            event_type="npc_response",
            timestamp=int(time.time() * 1000),
            text=text,
            language=language,
            source_event=source_event,
            subtitle_duration=SUBTITLE_DURATION,
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NpcState:
    message_type: str
    npc_is_thinking: bool
    npc_is_speaking: bool
    npc_emotion: str
    npc_behavior_state: str
    npc_emotion_intensity: float

    @classmethod
    def create(
        cls,
        is_thinking=False,
        is_speaking=False,
        emotion="neutral",
        behavior_state="idle",
        emotion_intensity=0.0,
    ):
        return cls(
            "npc_state",
            bool(is_thinking),
            bool(is_speaking),
            str(emotion),
            str(behavior_state),
            max(0.0, min(1.0, float(emotion_intensity))),
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NpcActionEvent:
    message_type: str
    event_type: str
    timestamp: int
    action: str
    emotion: str

    @classmethod
    def create(cls, action, emotion):
        return cls(
            message_type="interaction_event",
            event_type="npc_action",
            timestamp=int(time.time() * 1000),
            action=str(action),
            emotion=str(emotion),
        )

    def to_dict(self):
        return asdict(self)
