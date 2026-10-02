import cv2
import mediapipe as mp


def main():
    """打开摄像头并实时检测人体姿态。"""
    # MediaPipe 提供的人体姿态模型和绘图工具。
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # 0 表示电脑的默认摄像头。
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("无法打开摄像头，请检查设备连接或摄像头权限。")
        return

    try:
        # 使用 with 可以在程序结束时自动释放 Pose 模型。
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                success, frame = camera.read()
                if not success:
                    print("无法读取摄像头画面。")
                    break

                # OpenCV 使用 BGR，MediaPipe 需要 RGB 格式。
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测当前画面中的人体关键点。
                result = pose.process(rgb_frame)

                # 检测到人体后，绘制关键点和骨架连接。
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                # 实时显示姿态检测结果。
                cv2.imshow("Pose Detection", frame)

                # 按 q 键退出程序。
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # 释放摄像头并关闭所有 OpenCV 窗口。
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
