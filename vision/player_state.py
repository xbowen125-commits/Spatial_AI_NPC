from dataclasses import asdict, dataclass, replace

import cv2
import mediapipe as mp

from perception_config import (
    ATTENTION_EMA_ALPHA,
    ATTENTION_ENTER_THRESHOLD,
    ATTENTION_EXIT_THRESHOLD,
    ATTENTION_PITCH_MAX_DEGREES,
    ATTENTION_STABILITY_FRAMES,
    ATTENTION_WEIGHT_DISTANCE,
    ATTENTION_WEIGHT_FACE,
    ATTENTION_WEIGHT_PITCH,
    ATTENTION_WEIGHT_POSITION,
    ATTENTION_WEIGHT_STABILITY,
    ATTENTION_WEIGHT_YAW,
    ATTENTION_YAW_MAX_DEGREES,
    DISTANCE_CONFIRM_FRAMES,
    DISTANCE_ENTER_FAR,
    DISTANCE_ENTER_NEAR,
    DISTANCE_EXIT_FAR,
    DISTANCE_EXIT_NEAR,
    EYE_CONTACT_CONFIRM_FRAMES,
    EYE_CONTACT_ENTER_THRESHOLD,
    EYE_CONTACT_EXIT_THRESHOLD,
    EYE_CONTACT_WEIGHT_ATTENTION,
    EYE_CONTACT_WEIGHT_GAZE,
    FACE_CONFIRM_FRAMES,
    GAZE_CONFIRM_FRAMES,
    GAZE_HORIZONTAL_MAX_OFFSET,
    GAZE_HORIZONTAL_THRESHOLD,
    GAZE_IRIS_EMA_ALPHA,
    GAZE_ORIENTATION_GATE_BASE,
    GAZE_SCORE_EMA_ALPHA,
    GAZE_VERTICAL_MAX_OFFSET,
    GAZE_VERTICAL_THRESHOLD,
    GAZE_WEIGHT_CONSISTENCY,
    GAZE_WEIGHT_HORIZONTAL,
    GAZE_WEIGHT_RELIABILITY,
    GAZE_WEIGHT_VERTICAL,
    HAND_CONFIRM_FRAMES,
    HEAD_CONFIRM_FRAMES,
    HEAD_DIRECTION_PITCH_DEGREES,
    HEAD_DIRECTION_YAW_DEGREES,
    HEAD_POSE_EMA_ALPHA,
    LANDMARK_VISIBILITY_THRESHOLD,
    LOOKING_CONFIRM_FRAMES,
    POSE_HEAD_DOWN_HEIGHT_RATIO,
    POSE_HEAD_UP_HEIGHT_RATIO,
    POSE_HEAD_YAW_RATIO_THRESHOLD,
    POSITION_CONFIRM_FRAMES,
    POSITION_ENTER_LEFT,
    POSITION_ENTER_RIGHT,
    POSITION_EXIT_LEFT,
    POSITION_EXIT_RIGHT,
    PRESENCE_CONFIRM_FRAMES,
)


@dataclass(frozen=True)
class PlayerState:
    """统一描述视觉模块当前观察到的玩家状态。"""

    person_detected: bool
    face_detected: bool
    horizontal_position: str
    distance: str
    left_hand: str
    right_hand: str
    head_direction: str
    head_yaw: float
    head_pitch: float
    attention_score: float
    looking_at_npc: str
    gaze_horizontal: str
    gaze_vertical: str
    gaze_score: float
    eye_contact: str
    is_speaking: bool
    directed_speech: str

    @classmethod
    def no_person(cls):
        """创建未检测到玩家时的默认状态。"""
        return cls(
            False,
            False,
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            0.0,
            0.0,
            0.0,
            "unknown",
            "unknown",
            "unknown",
            0.0,
            "unknown",
            False,
            "false",
        )

    def to_dict(self):
        """返回标准 Player State 字典。"""
        return asdict(self)

    def to_transport_dict(self):
        """返回 UDP 数据；附带旧字段以兼容已有 Unity 代码。"""
        data = self.to_dict()
        data["message_type"] = "player_state"
        data["gesture"] = self.legacy_gesture
        data["position"] = self.horizontal_position.title()
        return data

    @property
    def legacy_gesture(self):
        if not self.person_detected:
            return "No person detected"
        if self.left_hand == "raised" and self.right_hand == "raised":
            return "Both hands raised"
        if self.left_hand == "raised":
            return "Left hand raised"
        if self.right_hand == "raised":
            return "Right hand raised"
        if self.left_hand == "down" and self.right_hand == "down":
            return "No hand raised"
        return "Unknown"


class PlayerStateTracker:
    """组合迟滞、连续帧确认，以及头部、注意力和视线平滑。"""

    _confirm_frames = {
        "person_detected": PRESENCE_CONFIRM_FRAMES,
        "face_detected": FACE_CONFIRM_FRAMES,
        "horizontal_position": POSITION_CONFIRM_FRAMES,
        "distance": DISTANCE_CONFIRM_FRAMES,
        "left_hand": HAND_CONFIRM_FRAMES,
        "right_hand": HAND_CONFIRM_FRAMES,
        "head_direction": HEAD_CONFIRM_FRAMES,
        "looking_at_npc": LOOKING_CONFIRM_FRAMES,
        "gaze_horizontal": GAZE_CONFIRM_FRAMES,
        "gaze_vertical": GAZE_CONFIRM_FRAMES,
        "eye_contact": EYE_CONTACT_CONFIRM_FRAMES,
    }

    def __init__(self):
        self.stable_state = PlayerState.no_person()
        self._candidate_values = {}
        self._candidate_counts = {}
        self._smoothed_head_yaw = None
        self._smoothed_head_pitch = None
        self._smoothed_attention = 0.0
        self._face_visible_frames = 0
        self._smoothed_gaze_x = None
        self._smoothed_gaze_y = None
        self._smoothed_gaze_score = 0.0

    def update(
        self,
        pose_landmarks,
        head_pose_observation=None,
        gaze_observation=None,
    ):
        """接收一帧 Pose、Head Pose 与 Gaze，返回稳定 Player State。"""
        pose_state = build_player_state(pose_landmarks, self.stable_state)
        face_detected = bool(
            head_pose_observation is not None
            and head_pose_observation.face_detected
        )
        head_pose_valid = bool(
            face_detected
            and head_pose_observation.head_yaw is not None
            and head_pose_observation.head_pitch is not None
        )

        if face_detected:
            self._face_visible_frames += 1
        else:
            self._face_visible_frames = 0

        if head_pose_valid:
            self._smoothed_head_yaw = _ema(
                self._smoothed_head_yaw,
                head_pose_observation.head_yaw,
                HEAD_POSE_EMA_ALPHA,
            )
            self._smoothed_head_pitch = _ema(
                self._smoothed_head_pitch,
                head_pose_observation.head_pitch,
                HEAD_POSE_EMA_ALPHA,
            )

        if head_pose_valid:
            raw_attention = calculate_attention_score(
                face_detected=True,
                head_yaw=self._smoothed_head_yaw,
                head_pitch=self._smoothed_head_pitch,
                horizontal_position=pose_state.horizontal_position,
                distance=pose_state.distance,
                face_visible_frames=self._face_visible_frames,
            )
            self._smoothed_attention = _ema(
                self._smoothed_attention,
                raw_attention,
                ATTENTION_EMA_ALPHA,
            )
            head_direction = head_direction_from_angles(
                self._smoothed_head_yaw,
                self._smoothed_head_pitch,
            )
        else:
            self._smoothed_attention = 0.0
            head_direction = pose_state.head_direction

        looking_at_npc = estimate_looking_at_npc(
            self._smoothed_attention,
            head_pose_valid,
            self.stable_state.looking_at_npc,
        )

        gaze_valid = bool(
            gaze_observation is not None and gaze_observation.valid
        )
        if gaze_valid:
            self._smoothed_gaze_x = _ema(
                self._smoothed_gaze_x,
                gaze_observation.horizontal_offset,
                GAZE_IRIS_EMA_ALPHA,
            )
            self._smoothed_gaze_y = _ema(
                self._smoothed_gaze_y,
                gaze_observation.vertical_offset,
                GAZE_IRIS_EMA_ALPHA,
            )
            raw_gaze_score = calculate_gaze_score(
                self._smoothed_gaze_x,
                self._smoothed_gaze_y,
                gaze_observation.consistency_score,
                gaze_observation.reliability,
            )
            self._smoothed_gaze_score = _ema(
                self._smoothed_gaze_score,
                raw_gaze_score,
                GAZE_SCORE_EMA_ALPHA,
            )
            gaze_horizontal = classify_gaze_horizontal(self._smoothed_gaze_x)
            gaze_vertical = classify_gaze_vertical(self._smoothed_gaze_y)
        else:
            self._smoothed_gaze_score = 0.0
            gaze_horizontal = "unknown"
            gaze_vertical = "unknown"

        eye_contact = estimate_eye_contact(
            looking_at_npc,
            self._smoothed_attention,
            self._smoothed_gaze_score,
            gaze_valid,
            self.stable_state.eye_contact,
        )

        raw_values = pose_state.to_dict()
        raw_values.update(
            {
                "person_detected": pose_state.person_detected or face_detected,
                "face_detected": face_detected,
                "head_direction": head_direction,
                "looking_at_npc": looking_at_npc,
                "gaze_horizontal": gaze_horizontal,
                "gaze_vertical": gaze_vertical,
                "eye_contact": eye_contact,
            }
        )

        stable_values = self.stable_state.to_dict()
        for field_name, required_frames in self._confirm_frames.items():
            stable_values[field_name] = self._confirm_value(
                field_name,
                raw_values[field_name],
                stable_values[field_name],
                required_frames,
            )

        if stable_values["face_detected"] and self._smoothed_head_yaw is not None:
            stable_values["head_yaw"] = round(self._smoothed_head_yaw, 1)
            stable_values["head_pitch"] = round(self._smoothed_head_pitch, 1)
            stable_values["attention_score"] = round(
                max(0.0, min(1.0, self._smoothed_attention)),
                2,
            )
        else:
            stable_values["head_yaw"] = 0.0
            stable_values["head_pitch"] = 0.0
            stable_values["attention_score"] = 0.0
            stable_values["looking_at_npc"] = "unknown"

        if gaze_valid and stable_values["face_detected"]:
            stable_values["gaze_score"] = round(
                max(0.0, min(1.0, self._smoothed_gaze_score)),
                2,
            )
        elif not stable_values["face_detected"]:
            stable_values["gaze_horizontal"] = "unknown"
            stable_values["gaze_vertical"] = "unknown"
            stable_values["gaze_score"] = 0.0
            stable_values["eye_contact"] = "unknown"
        elif (
            stable_values["gaze_horizontal"] == "unknown"
            and stable_values["gaze_vertical"] == "unknown"
        ):
            # 连续多帧无可靠 Iris 后才清空，短暂遮挡不会立即闪烁。
            stable_values["gaze_score"] = 0.0
            stable_values["eye_contact"] = "unknown"

        new_state = PlayerState(**stable_values)
        if not new_state.person_detected:
            self._reset_continuous_state()
            new_state = PlayerState.no_person()

        self.stable_state = new_state
        return new_state

    def _confirm_value(self, field_name, raw_value, stable_value, required_frames):
        if raw_value == stable_value:
            self._candidate_values.pop(field_name, None)
            self._candidate_counts.pop(field_name, None)
            return stable_value

        if self._candidate_values.get(field_name) == raw_value:
            self._candidate_counts[field_name] += 1
        else:
            self._candidate_values[field_name] = raw_value
            self._candidate_counts[field_name] = 1

        if self._candidate_counts[field_name] >= required_frames:
            self._candidate_values.pop(field_name, None)
            self._candidate_counts.pop(field_name, None)
            return raw_value
        return stable_value

    def _reset_continuous_state(self):
        self._smoothed_head_yaw = None
        self._smoothed_head_pitch = None
        self._smoothed_attention = 0.0
        self._face_visible_frames = 0
        self._smoothed_gaze_x = None
        self._smoothed_gaze_y = None
        self._smoothed_gaze_score = 0.0


def _ema(previous_value, current_value, alpha):
    """指数移动平均：保留历史趋势，同时平滑当前测量噪声。"""
    if previous_value is None:
        return float(current_value)
    return previous_value * (1.0 - alpha) + float(current_value) * alpha


def calculate_attention_score(
    face_detected,
    head_yaw,
    head_pitch,
    horizontal_position,
    distance,
    face_visible_frames,
):
    """用可解释规则计算 0 到 1 的轻量注意力近似分数。"""
    if not face_detected or head_yaw is None or head_pitch is None:
        return 0.0

    face_score = 1.0
    yaw_score = max(0.0, 1.0 - abs(head_yaw) / ATTENTION_YAW_MAX_DEGREES)
    pitch_score = max(
        0.0,
        1.0 - abs(head_pitch) / ATTENTION_PITCH_MAX_DEGREES,
    )
    position_score = {
        "center": 1.0,
        "left": 0.7,
        "right": 0.7,
    }.get(horizontal_position, 0.0)
    distance_score = {
        "medium": 1.0,
        "near": 0.7,
        "far": 0.7,
    }.get(distance, 0.0)
    stability_score = min(
        face_visible_frames / ATTENTION_STABILITY_FRAMES,
        1.0,
    )

    weighted_score = (
        face_score * ATTENTION_WEIGHT_FACE
        + yaw_score * ATTENTION_WEIGHT_YAW
        + pitch_score * ATTENTION_WEIGHT_PITCH
        + position_score * ATTENTION_WEIGHT_POSITION
        + distance_score * ATTENTION_WEIGHT_DISTANCE
        + stability_score * ATTENTION_WEIGHT_STABILITY
    )

    # 注意力至少需要 yaw 与 pitch 都大致朝前。
    # 门控范围为 0.5~1.0，避免仅凭“检测到脸”得到过高分数。
    orientation_gate = 0.5 + 0.5 * min(yaw_score, pitch_score)
    score = weighted_score * orientation_gate
    return max(0.0, min(1.0, score))


def estimate_looking_at_npc(attention_score, head_pose_valid, previous_value):
    """使用迟滞阈值生成 true / false / unknown 三态结果。"""
    if not head_pose_valid:
        return "unknown"
    if previous_value == "true":
        return "false" if attention_score <= ATTENTION_EXIT_THRESHOLD else "true"
    if previous_value == "false":
        return "true" if attention_score >= ATTENTION_ENTER_THRESHOLD else "false"
    if attention_score >= ATTENTION_ENTER_THRESHOLD:
        return "true"
    if attention_score <= ATTENTION_EXIT_THRESHOLD:
        return "false"
    return "unknown"


def calculate_gaze_score(
    horizontal_offset,
    vertical_offset,
    consistency_score,
    reliability,
):
    """根据校准后的虹膜偏移、双眼一致性和可靠度计算 Gaze Score。"""
    horizontal_score = max(
        0.0,
        1.0 - abs(horizontal_offset) / GAZE_HORIZONTAL_MAX_OFFSET,
    )
    vertical_score = max(
        0.0,
        1.0 - abs(vertical_offset) / GAZE_VERTICAL_MAX_OFFSET,
    )
    weighted_score = (
        horizontal_score * GAZE_WEIGHT_HORIZONTAL
        + vertical_score * GAZE_WEIGHT_VERTICAL
        + consistency_score * GAZE_WEIGHT_CONSISTENCY
        + reliability * GAZE_WEIGHT_RELIABILITY
    )
    orientation_gate = GAZE_ORIENTATION_GATE_BASE + (
        1.0 - GAZE_ORIENTATION_GATE_BASE
    ) * min(horizontal_score, vertical_score)
    score = weighted_score * orientation_gate
    return max(0.0, min(1.0, score))


def classify_gaze_horizontal(horizontal_offset):
    if horizontal_offset <= -GAZE_HORIZONTAL_THRESHOLD:
        return "left"
    if horizontal_offset >= GAZE_HORIZONTAL_THRESHOLD:
        return "right"
    return "center"


def classify_gaze_vertical(vertical_offset):
    if vertical_offset <= -GAZE_VERTICAL_THRESHOLD:
        return "up"
    if vertical_offset >= GAZE_VERTICAL_THRESHOLD:
        return "down"
    return "center"


def estimate_eye_contact(
    looking_at_npc,
    attention_score,
    gaze_score,
    gaze_valid,
    previous_value,
):
    """组合头部注意力和眼睛方向，并使用迟滞生成 Eye Contact。"""
    if not gaze_valid:
        return "unknown"
    if looking_at_npc == "false":
        return "false"
    if looking_at_npc != "true":
        return "unknown"

    eye_contact_score = (
        attention_score * EYE_CONTACT_WEIGHT_ATTENTION
        + gaze_score * EYE_CONTACT_WEIGHT_GAZE
    )

    if previous_value == "true":
        return (
            "false"
            if eye_contact_score <= EYE_CONTACT_EXIT_THRESHOLD
            else "true"
        )
    if previous_value == "false":
        return (
            "true"
            if eye_contact_score >= EYE_CONTACT_ENTER_THRESHOLD
            else "false"
        )
    if eye_contact_score >= EYE_CONTACT_ENTER_THRESHOLD:
        return "true"
    if eye_contact_score <= EYE_CONTACT_EXIT_THRESHOLD:
        return "false"
    return "unknown"


def head_direction_from_angles(head_yaw, head_pitch):
    """把连续 Head Pose 角度转换为旧版离散 head_direction。"""
    if head_pitch >= HEAD_DIRECTION_PITCH_DEGREES:
        return "up"
    if head_pitch <= -HEAD_DIRECTION_PITCH_DEGREES:
        return "down"
    if head_yaw <= -HEAD_DIRECTION_YAW_DEGREES:
        return "left"
    if head_yaw >= HEAD_DIRECTION_YAW_DEGREES:
        return "right"
    return "center"


def _is_visible(landmark):
    return landmark.visibility >= LANDMARK_VISIBILITY_THRESHOLD


def get_body_center(landmarks):
    pose_landmark = mp.solutions.pose.PoseLandmark
    left_shoulder = landmarks[pose_landmark.LEFT_SHOULDER]
    right_shoulder = landmarks[pose_landmark.RIGHT_SHOULDER]
    if not _is_visible(left_shoulder) or not _is_visible(right_shoulder):
        return None

    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    left_hip = landmarks[pose_landmark.LEFT_HIP]
    right_hip = landmarks[pose_landmark.RIGHT_HIP]
    if _is_visible(left_hip) and _is_visible(right_hip):
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        return (
            (shoulder_center_x + hip_center_x) / 2,
            (shoulder_center_y + hip_center_y) / 2,
        )
    return shoulder_center_x, shoulder_center_y


def estimate_horizontal_position(landmarks, previous_position="unknown"):
    body_center = get_body_center(landmarks)
    if body_center is None:
        return "unknown"
    center_x, _ = body_center
    if previous_position == "left" and center_x < POSITION_EXIT_LEFT:
        return "left"
    if previous_position == "right" and center_x > POSITION_EXIT_RIGHT:
        return "right"
    if center_x < POSITION_ENTER_LEFT:
        return "left"
    if center_x > POSITION_ENTER_RIGHT:
        return "right"
    return "center"


def _calculate_body_scale(landmarks):
    pose_landmark = mp.solutions.pose.PoseLandmark
    left_shoulder = landmarks[pose_landmark.LEFT_SHOULDER]
    right_shoulder = landmarks[pose_landmark.RIGHT_SHOULDER]
    if not _is_visible(left_shoulder) or not _is_visible(right_shoulder):
        return None

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    body_scale = shoulder_width
    left_hip = landmarks[pose_landmark.LEFT_HIP]
    right_hip = landmarks[pose_landmark.RIGHT_HIP]
    if _is_visible(left_hip) and _is_visible(right_hip):
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        torso_height = abs(hip_center_y - shoulder_center_y)
        body_scale = shoulder_width * 0.7 + torso_height * 0.3
    return body_scale


def estimate_relative_distance(landmarks, previous_distance="unknown"):
    body_scale = _calculate_body_scale(landmarks)
    if body_scale is None:
        return "unknown"
    if previous_distance == "near" and body_scale > DISTANCE_EXIT_NEAR:
        return "near"
    if previous_distance == "far" and body_scale < DISTANCE_EXIT_FAR:
        return "far"
    if body_scale >= DISTANCE_ENTER_NEAR:
        return "near"
    if body_scale <= DISTANCE_ENTER_FAR:
        return "far"
    return "medium"


def estimate_hand_states(landmarks):
    pose_landmark = mp.solutions.pose.PoseLandmark

    def estimate_one_hand(shoulder_index, wrist_index):
        shoulder = landmarks[shoulder_index]
        wrist = landmarks[wrist_index]
        if not _is_visible(shoulder) or not _is_visible(wrist):
            return "unknown"
        return "raised" if wrist.y < shoulder.y else "down"

    return (
        estimate_one_hand(pose_landmark.LEFT_SHOULDER, pose_landmark.LEFT_WRIST),
        estimate_one_hand(pose_landmark.RIGHT_SHOULDER, pose_landmark.RIGHT_WRIST),
    )


def estimate_pose_head_direction(landmarks):
    """Face Mesh 不可用时，使用 Pose 点做兼容回退。"""
    pose_landmark = mp.solutions.pose.PoseLandmark
    nose = landmarks[pose_landmark.NOSE]
    left_eye = landmarks[pose_landmark.LEFT_EYE]
    right_eye = landmarks[pose_landmark.RIGHT_EYE]
    left_ear = landmarks[pose_landmark.LEFT_EAR]
    right_ear = landmarks[pose_landmark.RIGHT_EAR]
    left_shoulder = landmarks[pose_landmark.LEFT_SHOULDER]
    right_shoulder = landmarks[pose_landmark.RIGHT_SHOULDER]
    required = (nose, left_eye, right_eye, left_shoulder, right_shoulder)
    if not all(_is_visible(landmark) for landmark in required):
        return "unknown"

    eye_center_x = (left_eye.x + right_eye.x) / 2
    if _is_visible(left_ear) and _is_visible(right_ear):
        face_width = abs(left_ear.x - right_ear.x)
    else:
        face_width = abs(left_eye.x - right_eye.x) * 2
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if face_width < 0.01 or shoulder_width < 0.01:
        return "unknown"

    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    head_height_ratio = (shoulder_center_y - nose.y) / shoulder_width
    if head_height_ratio >= POSE_HEAD_UP_HEIGHT_RATIO:
        return "up"
    if head_height_ratio <= POSE_HEAD_DOWN_HEIGHT_RATIO:
        return "down"

    yaw_ratio = (nose.x - eye_center_x) / face_width
    if yaw_ratio <= -POSE_HEAD_YAW_RATIO_THRESHOLD:
        return "left"
    if yaw_ratio >= POSE_HEAD_YAW_RATIO_THRESHOLD:
        return "right"
    return "center"


def build_player_state(pose_landmarks, previous_state=None):
    """仅根据 Pose 构建基础状态；Face 与 Attention 由 Tracker 合并。"""
    if pose_landmarks is None:
        return PlayerState.no_person()
    if previous_state is None:
        previous_state = PlayerState.no_person()

    landmarks = pose_landmarks.landmark
    left_hand, right_hand = estimate_hand_states(landmarks)
    return PlayerState(
        person_detected=True,
        face_detected=False,
        horizontal_position=estimate_horizontal_position(
            landmarks,
            previous_state.horizontal_position,
        ),
        distance=estimate_relative_distance(
            landmarks,
            previous_state.distance,
        ),
        left_hand=left_hand,
        right_hand=right_hand,
        head_direction=estimate_pose_head_direction(landmarks),
        head_yaw=0.0,
        head_pitch=0.0,
        attention_score=0.0,
        looking_at_npc="unknown",
        gaze_horizontal="unknown",
        gaze_vertical="unknown",
        gaze_score=0.0,
        eye_contact="unknown",
        is_speaking=False,
        directed_speech="false",
    )


def estimate_directed_speech(player_state, is_speaking):
    """用说话状态与视觉注意力粗略判断语音是否可能指向 NPC。"""
    if not is_speaking:
        return "false"
    if not player_state.person_detected:
        return "unknown"
    if player_state.eye_contact == "true":
        return "true"
    if (
        player_state.eye_contact == "false"
        or player_state.looking_at_npc == "false"
    ):
        return "false"
    # 头部可能朝向 NPC，但虹膜暂时不可用时不强行判断为 true。
    return "unknown"


def with_audio_state(player_state, is_speaking):
    """在不改变视觉 Tracker 的情况下合并持续音频状态。"""
    return replace(
        player_state,
        is_speaking=bool(is_speaking),
        directed_speech=estimate_directed_speech(
            player_state,
            bool(is_speaking),
        ),
    )


def draw_player_state_debug(
    frame,
    player_state,
    pose_landmarks=None,
    calibration_status="DEFAULT",
    microphone_status="DISABLED",
    last_speech="",
):
    """显示稳定 Player State，保持信息紧凑。"""
    height, width = frame.shape[:2]
    for boundary in (POSITION_ENTER_LEFT, POSITION_ENTER_RIGHT):
        x = int(width * boundary)
        cv2.line(frame, (x, 0), (x, height), (255, 255, 0), 1)

    if pose_landmarks is not None:
        body_center = get_body_center(pose_landmarks.landmark)
        if body_center is not None:
            center_x = int(body_center[0] * width)
            center_y = int(body_center[1] * height)
            cv2.circle(frame, (center_x, center_y), 7, (0, 0, 255), -1)

    face_text = "YES" if player_state.face_detected else "NO"
    looking_text = {
        "true": "YES",
        "false": "NO",
    }.get(player_state.looking_at_npc, "UNKNOWN")
    eye_contact_text = {
        "true": "YES",
        "false": "NO",
    }.get(player_state.eye_contact, "UNKNOWN")

    left_lines = (
        f"Person: {'YES' if player_state.person_detected else 'NO'}",
        f"Face: {face_text}",
        f"Position: {player_state.horizontal_position.upper()}",
        f"Distance: {player_state.distance.upper()}",
        f"Head Yaw: {player_state.head_yaw:+.1f}",
        f"Head Pitch: {player_state.head_pitch:+.1f}",
        f"Attention: {player_state.attention_score:.2f}",
        f"Looking: {looking_text}",
    )
    right_lines = (
        f"Gaze H: {player_state.gaze_horizontal.upper()}",
        f"Gaze V: {player_state.gaze_vertical.upper()}",
        f"Gaze Score: {player_state.gaze_score:.2f}",
        f"Eye Contact: {eye_contact_text}",
        f"Mic: {microphone_status}",
        f"Speaking: {'YES' if player_state.is_speaking else 'NO'}",
        f"Directed Speech: {player_state.directed_speech.upper()}",
        f"Calibration: {calibration_status}",
        f"L Hand: {player_state.left_hand.upper()}",
        f"R Hand: {player_state.right_hand.upper()}",
    )

    if last_speech:
        right_lines += (f"Last Speech: {last_speech}",)

    columns = ((15, left_lines), (max(width // 2, 310), right_lines))
    for start_x, lines in columns:
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (start_x, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
