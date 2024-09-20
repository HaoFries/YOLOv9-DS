import cv2
import os
import time

# 设置文件夹路径
folder_path = 'E:/work/SR-YOLO/dataset/MOT20-03_481_bilinear_small_X4/'  # 替换为你的文件夹路径
frame_count = 481  # 最后帧的编号

# 设定过滤参数
max_area = 10000  # 最大面积
aspect_ratio_threshold = 1.5  # 宽高比阈值

for i in range(1, frame_count + 1):
    # 生成文件名
    filename = folder_path + f'{i:06d}.jpg'

    # 读取图像
    img = cv2.imread(filename)

    if img is None:
        print(f'未能加载图片: {filename}')
        continue

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 边缘检测
    edges = cv2.Canny(gray, 100, 200)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 处理轮廓
    for contour in contours:
        area = cv2.contourArea(contour)

        # 过滤掉面积太大的物体
        if area > max_area:
            continue

        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 计算宽高比
        aspect_ratio = float(w) / h if h != 0 else 0

        # 过滤掉宽高比太大的物体
        if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1 / aspect_ratio_threshold:
            continue

        # 在原图上绘制边界框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Contours', img)

    # 每秒一帧显示
    time.sleep(1)

    # 检查是否按下了 ESC 键
    if cv2.waitKey(1) == 27:  # ESC 键
        break

# 释放所有窗口
cv2.destroyAllWindows()