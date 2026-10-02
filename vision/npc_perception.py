import cv2
import mediapipe as mp

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
)
from player_state import PlayerStateTracker, draw_player_state_debug


def main():
    """实时构建并显示包含 Attention 的稳定 Player State。"""
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("无法打开摄像头，请检查设备连接或摄像头权限。")
        return

    last_state = None
    state_tracker = PlayerStateTracker()
    gaze_calibrator = GazeCalibrator()

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
                player_state = state_tracker.update(
                    pose_result.pose_landmarks,
                    head_pose,
                    gaze,
                )

                if pose_result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                draw_face_debug(frame, face_landmarks, head_pose)
                draw_iris_debug(frame, gaze)
                draw_player_state_debug(
                    frame,
                    player_state,
                    pose_result.pose_landmarks,
                    gaze_calibrator.status,
                )

                if player_state != last_state:
                    print(player_state.to_dict())
                    last_state = player_state

                cv2.imshow("Spatial Perception v0.4", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    gaze_calibrator.start()
                    print("开始 Gaze 校准，请持续正视摄像头。")
                if key == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
