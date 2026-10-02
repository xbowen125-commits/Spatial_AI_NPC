from dataclasses import dataclass

import cv2
import numpy as np

from perception_config import HEAD_POSE_PITCH_SIGN, HEAD_POSE_YAW_SIGN


# MediaPipe Face Mesh 中用于简化 3D Head Pose 的关键点编号。
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
HEAD_POSE_LANDMARKS = (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    LEFT_MOUTH,
    RIGHT_MOUTH,
)

# 一个简化的通用人脸 3D 模型，使用OpenCV相机坐标：
# X向右、Y向下、Z远离相机。旧模型的Y/Z方向相反，会让正脸被表示成
# 接近绕X轴180度，从而产生 pitch=+/-177 这类Euler wrap。
FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, 63.6, 12.5),
        (-43.3, -32.7, 26.0),
        (43.3, -32.7, 26.0),
        (-28.9, 28.9, 24.1),
        (28.9, 28.9, 24.1),
    ],
    dtype=np.float64,
)


@dataclass
class HeadPoseObservation:
    """一帧 Face Mesh 产生的原始人脸与头姿结果。"""

    face_detected: bool
    head_yaw: float | None = None
    head_pitch: float | None = None
    nose_point: tuple[int, int] | None = None


def estimate_head_pose(face_landmarks, frame_width, frame_height):
    """使用 Face Mesh 关键点与 solvePnP 估计 yaw 和 pitch。"""
    if face_landmarks is None:
        return HeadPoseObservation(face_detected=False)

    image_points = []
    for landmark_index in HEAD_POSE_LANDMARKS:
        landmark = face_landmarks.landmark[landmark_index]
        image_points.append(
            (landmark.x * frame_width, landmark.y * frame_height)
        )
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = float(frame_width)
    camera_matrix = np.array(
        [
            (focal_length, 0.0, frame_width / 2),
            (0.0, focal_length, frame_height / 2),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    nose_point = tuple(image_points[0].astype(int))
    if not success:
        return HeadPoseObservation(True, nose_point=nose_point)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch = float(angles[0]) * HEAD_POSE_PITCH_SIGN
    yaw = float(angles[1]) * HEAD_POSE_YAW_SIGN

    return HeadPoseObservation(
        face_detected=True,
        head_yaw=yaw,
        head_pitch=pitch,
        nose_point=nose_point,
    )


def draw_face_debug(frame, face_landmarks, observation):
    """只绘制关键 Head Pose 点和一条方向线，避免画面过于复杂。"""
    if face_landmarks is None or not observation.face_detected:
        return

    height, width = frame.shape[:2]
    for landmark_index in HEAD_POSE_LANDMARKS:
        landmark = face_landmarks.landmark[landmark_index]
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 2, (255, 180, 0), -1)

    if (
        observation.nose_point is None
        or observation.head_yaw is None
        or observation.head_pitch is None
    ):
        return

    # 方向线仅用于调试：水平对应 yaw，垂直对应 pitch。
    end_point = (
        int(observation.nose_point[0] + observation.head_yaw * 2),
        int(observation.nose_point[1] - observation.head_pitch * 2),
    )
    cv2.line(frame, observation.nose_point, end_point, (0, 165, 255), 2)
