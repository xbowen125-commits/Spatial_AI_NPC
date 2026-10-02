from dataclasses import dataclass

import cv2

from perception_config import (
    GAZE_CALIBRATION_FRAMES,
    GAZE_DEFAULT_NEUTRAL_X,
    GAZE_DEFAULT_NEUTRAL_Y,
    GAZE_INVERT_HORIZONTAL,
    GAZE_INVERT_VERTICAL,
    GAZE_MAX_HORIZONTAL_DISAGREEMENT,
    GAZE_MAX_VERTICAL_DISAGREEMENT,
    GAZE_MIN_EYE_HEIGHT_PIXELS,
    GAZE_MIN_EYE_WIDTH_PIXELS,
    GAZE_RELIABLE_EYE_HEIGHT_PIXELS,
    GAZE_RELIABLE_EYE_WIDTH_PIXELS,
    GAZE_SINGLE_EYE_RELIABILITY_FACTOR,
)


# refine_landmarks=True 时可用的 MediaPipe Face Mesh 虹膜点。
LEFT_IRIS = (473, 474, 475, 476, 477)
RIGHT_IRIS = (468, 469, 470, 471, 472)

# 每只眼使用两个眼角与上下眼睑点建立局部坐标。
LEFT_EYE_CORNERS = (362, 263)
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_CORNERS = (33, 133)
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


@dataclass
class EyeMeasurement:
    valid: bool
    ratio_x: float = 0.0
    ratio_y: float = 0.0
    reliability: float = 0.0
    iris_center: tuple[int, int] | None = None


@dataclass
class RawGazeObservation:
    valid: bool
    iris_ratio_x: float = 0.0
    iris_ratio_y: float = 0.0
    consistency_score: float = 0.0
    reliability: float = 0.0
    iris_centers: tuple[tuple[int, int], ...] = ()


@dataclass
class GazeObservation:
    valid: bool
    horizontal_offset: float = 0.0
    vertical_offset: float = 0.0
    consistency_score: float = 0.0
    reliability: float = 0.0
    iris_centers: tuple[tuple[int, int], ...] = ()


class GazeCalibrator:
    """在当前运行期间采集用户正视摄像头时的中性虹膜比例。"""

    def __init__(self):
        self.neutral_x = GAZE_DEFAULT_NEUTRAL_X
        self.neutral_y = GAZE_DEFAULT_NEUTRAL_Y
        self.calibrated = False
        self._calibrating = False
        self._samples_x = []
        self._samples_y = []

    def start(self):
        """开始新的内存校准；用户应持续正视摄像头。"""
        self._calibrating = True
        self._samples_x.clear()
        self._samples_y.clear()

    def apply(self, raw_observation):
        """采集校准样本，并输出相对中性视线的偏移。"""
        if not raw_observation.valid:
            return GazeObservation(
                valid=False,
                iris_centers=raw_observation.iris_centers,
            )

        if self._calibrating:
            self._samples_x.append(raw_observation.iris_ratio_x)
            self._samples_y.append(raw_observation.iris_ratio_y)

            if len(self._samples_x) >= GAZE_CALIBRATION_FRAMES:
                self.neutral_x = sum(self._samples_x) / len(self._samples_x)
                self.neutral_y = sum(self._samples_y) / len(self._samples_y)
                self.calibrated = True
                self._calibrating = False

        horizontal_offset = raw_observation.iris_ratio_x - self.neutral_x
        vertical_offset = raw_observation.iris_ratio_y - self.neutral_y

        if GAZE_INVERT_HORIZONTAL:
            horizontal_offset = -horizontal_offset
        if GAZE_INVERT_VERTICAL:
            vertical_offset = -vertical_offset

        return GazeObservation(
            valid=True,
            horizontal_offset=horizontal_offset,
            vertical_offset=vertical_offset,
            consistency_score=raw_observation.consistency_score,
            reliability=raw_observation.reliability,
            iris_centers=raw_observation.iris_centers,
        )

    @property
    def status(self):
        if self._calibrating:
            return f"CALIBRATING {len(self._samples_x)}/{GAZE_CALIBRATION_FRAMES}"
        return "CALIBRATED" if self.calibrated else "DEFAULT"


def estimate_raw_gaze(face_landmarks, frame_width, frame_height):
    """计算虹膜在左右眼局部区域中的归一化位置，并融合双眼。"""
    if face_landmarks is None or len(face_landmarks.landmark) < 478:
        return RawGazeObservation(valid=False)

    left_eye = _measure_eye(
        face_landmarks,
        LEFT_IRIS,
        LEFT_EYE_CORNERS,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,
        frame_width,
        frame_height,
    )
    right_eye = _measure_eye(
        face_landmarks,
        RIGHT_IRIS,
        RIGHT_EYE_CORNERS,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,
        frame_width,
        frame_height,
    )

    valid_eyes = [eye for eye in (left_eye, right_eye) if eye.valid]
    iris_centers = tuple(
        eye.iris_center for eye in valid_eyes if eye.iris_center is not None
    )

    if not valid_eyes:
        return RawGazeObservation(False, iris_centers=iris_centers)

    if len(valid_eyes) == 2:
        horizontal_difference = abs(left_eye.ratio_x - right_eye.ratio_x)
        vertical_difference = abs(left_eye.ratio_y - right_eye.ratio_y)

        # 双眼严重冲突时不强行输出方向。
        if (
            horizontal_difference > GAZE_MAX_HORIZONTAL_DISAGREEMENT
            or vertical_difference > GAZE_MAX_VERTICAL_DISAGREEMENT
        ):
            return RawGazeObservation(False, iris_centers=iris_centers)

        consistency_score = 1.0 - 0.5 * (
            horizontal_difference / GAZE_MAX_HORIZONTAL_DISAGREEMENT
            + vertical_difference / GAZE_MAX_VERTICAL_DISAGREEMENT
        )
        reliability = (left_eye.reliability + right_eye.reliability) / 2
    else:
        consistency_score = 0.5
        reliability = (
            valid_eyes[0].reliability * GAZE_SINGLE_EYE_RELIABILITY_FACTOR
        )

    total_weight = sum(max(eye.reliability, 0.01) for eye in valid_eyes)
    ratio_x = sum(
        eye.ratio_x * max(eye.reliability, 0.01) for eye in valid_eyes
    ) / total_weight
    ratio_y = sum(
        eye.ratio_y * max(eye.reliability, 0.01) for eye in valid_eyes
    ) / total_weight

    return RawGazeObservation(
        valid=True,
        iris_ratio_x=ratio_x,
        iris_ratio_y=ratio_y,
        consistency_score=max(0.0, min(1.0, consistency_score)),
        reliability=max(0.0, min(1.0, reliability)),
        iris_centers=iris_centers,
    )


def _measure_eye(
    face_landmarks,
    iris_indices,
    corner_indices,
    top_index,
    bottom_index,
    frame_width,
    frame_height,
):
    landmarks = face_landmarks.landmark
    corner_a = landmarks[corner_indices[0]]
    corner_b = landmarks[corner_indices[1]]
    top = landmarks[top_index]
    bottom = landmarks[bottom_index]

    iris_x = sum(landmarks[index].x for index in iris_indices) / len(iris_indices)
    iris_y = sum(landmarks[index].y for index in iris_indices) / len(iris_indices)
    min_x, max_x = sorted((corner_a.x, corner_b.x))
    min_y, max_y = sorted((top.y, bottom.y))
    eye_width_pixels = (max_x - min_x) * frame_width
    eye_height_pixels = (max_y - min_y) * frame_height
    iris_center = (int(iris_x * frame_width), int(iris_y * frame_height))

    if (
        eye_width_pixels < GAZE_MIN_EYE_WIDTH_PIXELS
        or eye_height_pixels < GAZE_MIN_EYE_HEIGHT_PIXELS
    ):
        return EyeMeasurement(False, iris_center=iris_center)

    ratio_x = (iris_x - min_x) / max(max_x - min_x, 1e-6)
    ratio_y = (iris_y - min_y) / max(max_y - min_y, 1e-6)

    # 略微允许虹膜落在轮廓外，兼容侧视时的透视误差。
    if not (-0.15 <= ratio_x <= 1.15 and -0.20 <= ratio_y <= 1.20):
        return EyeMeasurement(False, iris_center=iris_center)

    width_score = min(
        eye_width_pixels / GAZE_RELIABLE_EYE_WIDTH_PIXELS,
        1.0,
    )
    height_score = min(
        eye_height_pixels / GAZE_RELIABLE_EYE_HEIGHT_PIXELS,
        1.0,
    )
    reliability = width_score * height_score

    return EyeMeasurement(
        valid=True,
        ratio_x=ratio_x,
        ratio_y=ratio_y,
        reliability=reliability,
        iris_center=iris_center,
    )


def draw_iris_debug(frame, gaze_observation):
    """只绘制左右虹膜中心，不绘制完整 Face Mesh。"""
    color = (0, 255, 255) if gaze_observation.valid else (0, 128, 255)
    for iris_center in gaze_observation.iris_centers:
        cv2.circle(frame, iris_center, 3, color, -1)
