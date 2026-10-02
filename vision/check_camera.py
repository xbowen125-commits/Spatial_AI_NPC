import cv2


print("开始检测摄像头...")

for i in range(10):
    camera = cv2.VideoCapture(i)

    if camera.isOpened():
        print(f"发现摄像头编号: {i}")
        camera.release()
    else:
        print(f"编号 {i} 不可用")
