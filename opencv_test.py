import cv2

cap = cv2.VideoCapture(1)  # 调用目录下的视频
# cap=cv2.VideoCapture(0)  #调用摄像头‘0’一般是打开电脑自带摄像头，‘1’是打开外部摄像头（只有一个摄像头的情况）

if False == cap.isOpened():
    print(0)
else:
    print(1)
