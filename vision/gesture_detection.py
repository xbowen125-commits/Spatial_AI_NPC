import cv2
import mediapipe as mp

from player_state import estimate_hand_states


def detect_raised_hands(landmarks):
    """根据手腕和肩膀的高度，判断用户是否举手。"""
    left_hand, right_hand = estimate_hand_states(landmarks)

    if left_hand == "raised" and right_hand == "raised":
        return "Both hands raised"
    if left_hand == "raised":
        return "Left hand raised"
    if right_hand == "raised":
        return "Right hand raised"
    if left_hand == "unknown" or right_hand == "unknown":
        return "Unknown"
    return "No hand raised"


def main():
    """实时检测人体姿态，并识别简单的举手动作。"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("无法打开摄像头，请检查设备连接或摄像头权限。")
        return

    try:
        # 使用轻量级姿态模型，适合入门阶段的实时实验。
        with mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                success, frame = camera.read()
                if not success:
                    print("无法读取摄像头画面。")
                    break

                # MediaPipe 需要 RGB 图像，OpenCV 默认提供 BGR 图像。
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)

                gesture = "No person detected"

                if result.pose_landmarks:
                    # 绘制人体关键点和骨架连接。
                    mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    # 根据关键点位置判断举手动作。
                    gesture = detect_raised_hands(result.pose_landmarks.landmark)

                # OpenCV 默认字体不支持中文，因此画面提示使用简单英文。
                cv2.putText(
                    frame,
                    gesture,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Gesture Detection", frame)

                # 按 q 键退出程序。
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
