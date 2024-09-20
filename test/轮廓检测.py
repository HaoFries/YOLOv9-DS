# import cv2
# import os
# import time
#
# # 设置文件夹路径
# folder_path = 'E:/work/SR-YOLO/dataset/MOT20-03_481_bilinear_small_X4/'  # 替换为你的文件夹路径
# frame_count = 481  # 最后帧的编号
#
# for i in range(1, frame_count + 1):
#     # 生成文件名
#     filename = folder_path + f'{i:06d}.jpg'
#
#     # 读取图像
#     img = cv2.imread(filename)
#
#     if img is None:
#         print(f'未能加载图片: {filename}')
#         continue
#
#     # 转换为灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 使用 Canny 边缘检测
#     edges = cv2.Canny(gray, 100, 200)
#
#     # 寻找轮廓
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 在原图上绘制轮廓
#     cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
#
#     # 显示图像
#     cv2.imshow('Contours', img)
#
#     # 每秒一帧显示
#     time.sleep(1)
#
#     # 检查是否按下了 ESC 键
#     if cv2.waitKey(1) == 27:  # ESC 键
#         break
#
# # 释放所有窗口
# cv2.destroyAllWindows()
import cv2
import os
import time

# 设置文件夹路径
input_folder_path = 'E:/work/SR-YOLO/dataset/MOT20-03_481_bilinear_small_X4/'  # 输入文件夹路径
edges_output_folder = 'E:/work/small_person_AP/edges_output/'  # 边缘输出文件夹
contours_output_folder = 'E:/work/small_person_AP/contours_output/'  # 轮廓输出文件夹
frame_count = 481  # 最后帧的编号

# 创建输出文件夹（如果不存在）
os.makedirs(edges_output_folder, exist_ok=True)
os.makedirs(contours_output_folder, exist_ok=True)

for i in range(1, frame_count + 1):
    # 生成文件名
    filename = input_folder_path + f'{i:06d}.jpg'

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

    # 在原图上绘制轮廓
    contours_img = img.copy()  # 创建一个副本用于绘制轮廓
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 1)

    # 保存边缘图像
    edges_output_filename = edges_output_folder + f'{i:06d}.jpg'
    cv2.imwrite(edges_output_filename, edges)

    # 保存轮廓图像
    contours_output_filename = contours_output_folder + f'{i:06d}.jpg'
    cv2.imwrite(contours_output_filename, contours_img)

    # 显示图像
    cv2.imshow('Contours', contours_img)

    # 每秒一帧显示
    time.sleep(1)

    # 检查是否按下了 ESC 键
    if cv2.waitKey(1) == 27:  # ESC 键
        break

# 释放所有窗口
cv2.destroyAllWindows()