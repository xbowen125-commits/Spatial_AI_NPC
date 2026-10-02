import cv2
import mediapipe as mp

from player_state import estimate_horizontal_position, estimate_relative_distance


def detect_person_position(landmarks):
    """根据双肩关键点，判断人物的水平位置和大致距离。"""
    position = estimate_horizontal_position(landmarks)
    distance = estimate_relative_distance(landmarks)
    return position.title(), distance.title()


def main():
    """实时检测人物在画面中的位置和大致距离。"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("无法打开摄像头，请检查设备连接或摄像头权限。")
        return

    try:
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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)

                position_text = "No person detected"

                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    position, distance = detect_person_position(
                        result.pose_landmarks.landmark
                    )
                    position_text = f"Position: {position} | Distance: {distance}"

                height, width = frame.shape[:2]

                # 画出左、中、右区域的参考分界线。
                cv2.line(frame, (int(width * 0.4), 0), (int(width * 0.4), height), (255, 255, 0), 1)
                cv2.line(frame, (int(width * 0.6), 0), (int(width * 0.6), height), (255, 255, 0), 1)

                cv2.putText(
                    frame,
                    position_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Person Position", frame)

                # 按 q 键退出程序。
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
