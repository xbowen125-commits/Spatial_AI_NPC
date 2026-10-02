import cv2


def main():
    """打开默认摄像头并实时显示画面。"""
    # 0 表示电脑的默认摄像头。
    camera = cv2.VideoCapture(0)

    # 如果摄像头无法打开，给出提示并结束程序。
    if not camera.isOpened():
        print("无法打开摄像头，请检查设备连接或摄像头权限。")
        return

    try:
        while True:
            # 从摄像头读取一帧画面。
            success, frame = camera.read()
            if not success:
                print("无法读取摄像头画面。")
                break

            # 实时显示当前画面。
            cv2.imshow("Camera Test", frame)

            # 按 q 键退出摄像头测试。
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 无论程序如何结束，都释放摄像头并关闭窗口。
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
