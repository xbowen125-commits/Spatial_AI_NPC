"""Head Pose坐标转换的轻量回归测试。"""

import unittest

import cv2
import numpy as np

from attention_detection import FACE_MODEL_POINTS, estimate_head_pose
from player_state import head_direction_from_angles


class _Landmark:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class _FaceLandmarks:
    def __init__(self, image_points, width, height):
        self.landmark = [_Landmark() for _ in range(468)]
        indices = (1, 152, 33, 263, 61, 291)
        for index, point in zip(indices, image_points):
            self.landmark[index] = _Landmark(
                float(point[0]) / width,
                float(point[1]) / height,
            )


def _synthetic_face(axis_angle_degrees=(0.0, 0.0, 0.0)):
    width, height = 640, 480
    camera_matrix = np.array(
        [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation_vector = np.radians(
        np.array(axis_angle_degrees, dtype=np.float64)
    ).reshape(3, 1)
    image_points, _ = cv2.projectPoints(
        FACE_MODEL_POINTS,
        rotation_vector,
        np.array([[0.0], [0.0], [1000.0]], dtype=np.float64),
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )
    return _FaceLandmarks(image_points.reshape(-1, 2), width, height), width, height


class HeadPoseTests(unittest.TestCase):
    def test_frontal_face_is_near_zero_not_180(self):
        landmarks, width, height = _synthetic_face()
        result = estimate_head_pose(landmarks, width, height)
        self.assertAlmostEqual(result.head_yaw, 0.0, delta=0.5)
        self.assertAlmostEqual(result.head_pitch, 0.0, delta=0.5)

    def test_small_rotation_stays_continuous(self):
        landmarks, width, height = _synthetic_face((10.0, 20.0, 0.0))
        result = estimate_head_pose(landmarks, width, height)
        self.assertLess(abs(result.head_pitch), 30.0)
        self.assertLess(abs(result.head_yaw), 30.0)

    def test_existing_direction_thresholds_are_unchanged(self):
        self.assertEqual(head_direction_from_angles(0.0, 0.0), "center")
        self.assertEqual(head_direction_from_angles(0.0, 13.0), "up")
        self.assertEqual(head_direction_from_angles(0.0, -13.0), "down")
        self.assertEqual(head_direction_from_angles(-16.0, 0.0), "left")
        self.assertEqual(head_direction_from_angles(16.0, 0.0), "right")


if __name__ == "__main__":
    unittest.main(verbosity=2)
