import json
import socket
import sys
import time
from pathlib import Path

# 允许从 vision/unity_bridge.py 直接运行时导入同级 audio 与 events 包。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mediapipe as mp

from audio.audio_config import HALF_DUPLEX_MODE, LAST_SPEECH_DISPLAY_LENGTH
from audio.audio_controller import AudioController
from audio.speech_output import SpeechOutputController
from behavior.behavior_manager import BehaviorManager
from behavior.event_detector import EventDetector
from emotion.emotion_config import (
    NPC_STATE_HEARTBEAT_SECONDS,
    NPC_STATE_INTENSITY_SYNC_THRESHOLD,
)
from emotion.emotion_engine import EmotionEngine
from emotion.emotion_policy import EmotionAwareBehaviorRules
from evaluation.interaction_logger import (
    BehaviorLogger,
    EmotionLogger,
    InteractionLogger,
)
from events.interaction_event import (
    InteractionEvent,
    NpcActionEvent,
    NpcState,
    ResponseEvent,
)
from memory.memory_policy import RelationshipBehaviorRules
from memory.memory_store import MemoryStore
from npc.agent import NpcAgent
from npc.world_context import build_world_context
from attention_detection import draw_face_debug, estimate_head_pose
from gaze_detection import (
    GazeCalibrator,
    draw_iris_debug,
    estimate_raw_gaze,
)
from perception_config import (
    FACE_MAX_COUNT,
    FACE_MIN_DETECTION_CONFIDENCE,
    FACE_MIN_TRACKING_CONFIDENCE,
    STATE_HEARTBEAT_SECONDS,
    UNITY_IP,
    UNITY_PORT,
)
from player_state import (
    PlayerStateTracker,
    draw_player_state_debug,
    with_audio_state,
)


def send_state(udp_socket, player_state):
    """把完整 Player State 转换为 JSON，并发送给 Unity。"""
    message = json.dumps(player_state.to_transport_dict()).encode("utf-8")
    udp_socket.sendto(message, (UNITY_IP, UNITY_PORT))
    print(message.decode("utf-8"))


def send_interaction_event(udp_socket, interaction_event):
    """发送一次性事件；不会混入 Player State 心跳。"""
    message = json.dumps(
        interaction_event.to_dict(),
        ensure_ascii=False,
    ).encode("utf-8")
    udp_socket.sendto(message, (UNITY_IP, UNITY_PORT))


def dispatch_behavior_decision(
    udp_socket,
    speech_output,
    behavior_manager,
    decision,
):
    """把 Behavior Decision 接入现有 Action 与 TTS，不改变 Unity 协议。"""
    send_interaction_event(
        udp_socket,
        NpcActionEvent.create(decision.action, decision.emotion),
    )
    if not decision.reply:
        return

    response_event = ResponseEvent.create(
        decision.reply,
        "zh",
        source_event="world_event",
    )
    if not speech_output.enqueue(response_event):
        send_interaction_event(udp_socket, response_event)
        behavior_manager.on_text_only_finished()
        print(f"NPC主动行为（文字）：{response_event.text}")


def main():
    """运行感知、异步 NPC Agent、本地 TTS 与 Unity 通信。"""
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("无法打开摄像头，请检查设备连接或摄像头权限。")
        return

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_sent_state = None
    last_sent_time = 0.0
    state_tracker = PlayerStateTracker()
    gaze_calibrator = GazeCalibrator()
    audio_controller = AudioController()
    audio_controller.start()
    speech_output = SpeechOutputController()
    speech_output.start()
    npc_agent = NpcAgent()
    npc_agent.start()
    interaction_logger = InteractionLogger()
    behavior_logger = BehaviorLogger()
    emotion_logger = EmotionLogger()
    emotion_engine = EmotionEngine(logger=emotion_logger)
    memory_store = MemoryStore()
    event_detector = EventDetector()
    behavior_manager = BehaviorManager(
        rules=RelationshipBehaviorRules(
            base_rules=EmotionAwareBehaviorRules(
                state_provider=lambda: emotion_engine.state,
            )
        ),
        logger=behavior_logger,
    )
    interaction_starts = {}
    interaction_stt_latencies = {}
    pending_tts_logs = {}
    npc_is_thinking = False
    npc_is_speaking = False
    last_npc_state = None
    last_npc_intensity = -1.0
    last_npc_state_sent_at = 0.0

    def publish_npc_state(force=False):
        nonlocal last_npc_state, last_npc_intensity, last_npc_state_sent_at
        emotion_state = emotion_engine.state
        state = NpcState.create(
            npc_is_thinking,
            npc_is_speaking,
            emotion_state.emotion,
            behavior_manager.state,
            emotion_state.intensity,
        )
        signature = (
            state.npc_is_thinking,
            state.npc_is_speaking,
            state.npc_emotion,
            state.npc_behavior_state,
        )
        current_time = time.monotonic()
        intensity_changed = (
            abs(state.npc_emotion_intensity - last_npc_intensity)
            >= NPC_STATE_INTENSITY_SYNC_THRESHOLD
        )
        heartbeat_due = (
            current_time - last_npc_state_sent_at
            >= NPC_STATE_HEARTBEAT_SECONDS
        )
        if (
            force
            or signature != last_npc_state
            or intensity_changed
            or heartbeat_due
        ):
            send_interaction_event(udp_socket, state)
            last_npc_state = signature
            last_npc_intensity = state.npc_emotion_intensity
            last_npc_state_sent_at = current_time

    publish_npc_state(force=True)
    pending_speech_state = None
    last_active_speech_state = None

    try:
        with mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose, mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=FACE_MAX_COUNT,
            refine_landmarks=True,
            min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_MIN_TRACKING_CONFIDENCE,
        ) as face_mesh:
            while True:
                success, frame = camera.read()
                if not success:
                    print("无法读取摄像头画面。")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb_frame)
                face_result = face_mesh.process(rgb_frame)
                face_landmarks = (
                    face_result.multi_face_landmarks[0]
                    if face_result.multi_face_landmarks
                    else None
                )

                height, width = frame.shape[:2]
                head_pose = estimate_head_pose(face_landmarks, width, height)
                raw_gaze = estimate_raw_gaze(face_landmarks, width, height)
                gaze = gaze_calibrator.apply(raw_gaze)
                visual_state = state_tracker.update(
                    pose_result.pose_landmarks,
                    head_pose,
                    gaze,
                )
                player_state = with_audio_state(
                    visual_state,
                    audio_controller.is_speaking,
                )
                emotion_engine.update()
                detected_world_events = event_detector.update(player_state)
                memory_store.observe_world_events(detected_world_events)
                for world_event in detected_world_events:
                    emotion_engine.process_world_event(
                        world_event,
                        memory_store.profile.relationship_level,
                    )
                speech_event_received = False
                if audio_controller.is_recording and player_state.is_speaking:
                    last_active_speech_state = player_state

                # 自动达到最长录音时间时结束；STT 始终在后台线程执行。
                if audio_controller.update():
                    pending_speech_state = last_active_speech_state or player_state

                speech_outcome = audio_controller.poll_result()
                if speech_outcome is not None:
                    if speech_outcome.result is not None:
                        event_state = pending_speech_state or player_state
                        memory_store.record_speech_interaction()
                        emotion_engine.process_speech_event(
                            memory_store.profile.relationship_level,
                        )
                        speech_event = InteractionEvent.speech(
                            speech_outcome.result,
                            event_state,
                            memory_store.profile.relationship_level,
                            emotion_engine.state.emotion,
                            emotion_engine.state.intensity,
                        )
                        interaction_starts[speech_event.timestamp] = (
                            time.monotonic()
                            - speech_outcome.latency_ms / 1000.0
                        )
                        interaction_stt_latencies[speech_event.timestamp] = (
                            speech_outcome.latency_ms
                        )
                        send_interaction_event(udp_socket, speech_event)
                        print(
                            f"识别结果 [{speech_outcome.result.language}]："
                            f"{speech_outcome.result.text}"
                        )
                        speech_event_received = True
                        behavior_manager.on_speech_event()
                        if not npc_agent.submit(speech_event):
                            print("该语音未通过 Directed Speech Gate，NPC 不回应。")
                            interaction_starts.pop(speech_event.timestamp, None)
                            interaction_stt_latencies.pop(
                                speech_event.timestamp,
                                None,
                            )
                    else:
                        print(f"语音识别失败，视觉系统继续运行：{speech_outcome.error}")
                    pending_speech_state = None
                    last_active_speech_state = None

                # Agent 调用在独立线程执行；主线程只处理生命周期和验证后决策。
                while True:
                    agent_event = npc_agent.poll_event()
                    if agent_event is None:
                        break
                    if agent_event.phase == "started":
                        npc_is_thinking = True
                        behavior_manager.on_agent_started()
                        publish_npc_state()
                        continue

                    npc_is_thinking = False
                    result = agent_event.result
                    decision = result.decision if result is not None else None
                    if decision is None:
                        behavior_manager.on_text_only_finished()
                        publish_npc_state()
                        interaction_starts.pop(
                            agent_event.speech_event.timestamp,
                            None,
                        )
                        interaction_stt_latencies.pop(
                            agent_event.speech_event.timestamp,
                            None,
                        )
                        continue

                    behavior_manager.on_response_generated()
                    publish_npc_state()
                    send_interaction_event(
                        udp_socket,
                        NpcActionEvent.create(decision.action, decision.emotion),
                    )
                    response_language = (
                        "zh"
                        if agent_event.speech_event.language.lower().startswith("zh")
                        else "en"
                    )
                    response_event = ResponseEvent.create(
                        decision.reply,
                        response_language,
                    )
                    speech_timestamp = agent_event.speech_event.timestamp
                    evaluation_data = (
                        agent_event.speech_event,
                        build_world_context(agent_event.speech_event),
                        result,
                        interaction_stt_latencies.pop(speech_timestamp, 0),
                        interaction_starts.pop(
                            speech_timestamp,
                            time.monotonic(),
                        ),
                    )
                    if not speech_output.enqueue(response_event):
                        # TTS 被关闭或不可用时，验证后的文字回复仍发送给 Unity。
                        send_interaction_event(udp_socket, response_event)
                        print(f"NPC（文字）：{response_event.text}")
                        interaction_logger.write(
                            *evaluation_data[:3],
                            stt_latency=evaluation_data[3],
                            tts_latency=0,
                            total_latency=int(
                                (time.monotonic() - evaluation_data[4]) * 1000
                            ),
                        )
                    else:
                        pending_tts_logs[response_event.timestamp] = evaluation_data

                # TTS 工作线程只汇报生命周期；UDP 始终由视觉主线程发送。
                while True:
                    output_event = speech_output.poll_event()
                    if output_event is None:
                        break
                    if output_event.phase == "started":
                        if HALF_DUPLEX_MODE:
                            audio_controller.set_input_suppressed(True)
                        send_interaction_event(
                            udp_socket,
                            output_event.response_event,
                        )
                        npc_is_speaking = True
                        behavior_manager.on_tts_started()
                        publish_npc_state()
                        print(f"NPC：{output_event.response_event.text}")
                    elif output_event.phase == "finished":
                        npc_is_speaking = False
                        behavior_manager.on_tts_finished()
                        publish_npc_state()
                        audio_controller.set_input_suppressed(False)
                        evaluation_data = pending_tts_logs.pop(
                            output_event.response_event.timestamp,
                            None,
                        )
                        if evaluation_data is not None:
                            interaction_logger.write(
                                *evaluation_data[:3],
                                stt_latency=evaluation_data[3],
                                tts_latency=output_event.latency_ms,
                                total_latency=int(
                                    (time.monotonic() - evaluation_data[4]) * 1000
                                ),
                            )
                    elif output_event.phase == "error_before_start":
                        send_interaction_event(
                            udp_socket,
                            output_event.response_event,
                        )
                        npc_is_speaking = False
                        behavior_manager.on_text_only_finished()
                        publish_npc_state()
                        audio_controller.set_input_suppressed(False)
                        print(
                            "TTS 不可用，已仅发送文字回复："
                            f"{output_event.error}"
                        )
                        evaluation_data = pending_tts_logs.pop(
                            output_event.response_event.timestamp,
                            None,
                        )
                        if evaluation_data is not None:
                            interaction_logger.write(
                                *evaluation_data[:3],
                                stt_latency=evaluation_data[3],
                                tts_latency=output_event.latency_ms,
                                total_latency=int(
                                    (time.monotonic() - evaluation_data[4]) * 1000
                                ),
                            )
                    elif output_event.phase == "error_after_start":
                        print(f"TTS 播放失败，系统继续运行：{output_event.error}")

                # 玩家语音优先级最高；同一帧检测到的主动事件直接取消。
                if speech_event_received:
                    behavior_manager.discard_events(detected_world_events)
                    proactive_decision = None
                else:
                    proactive_decision = behavior_manager.handle_events(
                        detected_world_events,
                    )
                    if proactive_decision is None:
                        proactive_decision = behavior_manager.tick()
                if proactive_decision is not None:
                    dispatch_behavior_decision(
                        udp_socket,
                        speech_output,
                        behavior_manager,
                        proactive_decision,
                    )
                behavior_manager.observe_player(event_detector.person_present)
                publish_npc_state()

                if pose_result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                current_time = time.monotonic()
                state_changed = player_state != last_sent_state
                heartbeat_due = (
                    current_time - last_sent_time >= STATE_HEARTBEAT_SECONDS
                )
                if state_changed or heartbeat_due:
                    send_state(udp_socket, player_state)
                    last_sent_state = player_state
                    last_sent_time = current_time

                draw_face_debug(frame, face_landmarks, head_pose)
                draw_iris_debug(frame, gaze)
                draw_player_state_debug(
                    frame,
                    player_state,
                    pose_result.pose_landmarks,
                    gaze_calibrator.status,
                    audio_controller.mic_status,
                    audio_controller.last_speech[:LAST_SPEECH_DISPLAY_LENGTH],
                )
                cv2.imshow("Proactive NPC Behavior v0.9", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    gaze_calibrator.start()
                    print("开始 Gaze 校准，请持续正视摄像头。")
                if key == ord(" "):
                    if HALF_DUPLEX_MODE and (
                        npc_is_thinking or speech_output.busy
                    ):
                        print("NPC 正在思考或说话，请等待结束后再录音。")
                        continue
                    was_recording = audio_controller.is_recording
                    audio_controller.toggle_recording()
                    if not was_recording and audio_controller.is_recording:
                        last_active_speech_state = None
                    if was_recording and not audio_controller.is_recording:
                        # 保存说话结束瞬间的视觉上下文，供稍后 STT 事件使用。
                        pending_speech_state = (
                            last_active_speech_state or player_state
                        )
                if key == ord("q"):
                    break
    finally:
        npc_agent.shutdown()
        speech_output.shutdown()
        try:
            send_interaction_event(
                udp_socket,
                NpcState.create(False, False, "neutral", "idle", 0.0),
            )
        except OSError:
            pass
        audio_controller.set_input_suppressed(False)
        audio_controller.close()
        camera.release()
        udp_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
