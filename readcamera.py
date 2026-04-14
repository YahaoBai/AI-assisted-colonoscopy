import cv2
import sys

# 设置相机设备索引为 2，对应 /dev/video2
device_index = 2
cap = cv2.VideoCapture(device_index)

if not cap.isOpened():
    print(f"Error: 无法打开设备 /dev/video{device_index}")
    sys.exit(1)

# 获取并打印基础参数
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"成功连接设备 /dev/video{device_index}")
print(f"当前采集参数: {int(width)}x{int(height)} @ {fps}FPS")
print("提示: 选中弹出的视频窗口，按下键盘上的小写字母 'q' 即可退出。")

# 进入实时读取循环
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    
    # 检查帧是否正确读取
    if not ret:
        print("Error: 无法获取图像帧，数据流可能已中断。")
        break

    # 在图形窗口中显示当前帧，窗口名称为 'USB Camera Stream'
    cv2.imshow('USB Camera Stream', frame)

    # cv2.waitKey(1) 作用有两点：
    # 1. 挂起当前线程 1 毫秒，允许 OpenCV 处理内部 GUI 绘制事件，从而刷新窗口图像。
    # 2. 捕获在这 1 毫秒内键盘的输入。
    # 0xFF 掩码用于处理跨平台的按键编码一致性问题。
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("收到退出指令，正在关闭...")
        break

# 释放硬件资源
cap.release()
# 销毁所有由 OpenCV 创建的 GUI 窗口
cv2.destroyAllWindows()