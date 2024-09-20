##实现了目标框精确定位到行人
import os
import cv2
from tqdm import tqdm
from collections import defaultdict


def draw_bounding_boxes_from_gt(gt_file, input_dir, output_dir):
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 用于存储每一帧的所有目标框
    frames_data = defaultdict(list)

    # 打开gt文件并读取内容
    with open(gt_file, 'r') as file:
        lines = file.readlines()

    # 遍历每一行，收集每一帧的目标框数据
    for line in lines:
        # 将每行数据按逗号分隔并转换为浮点数
        data = list(map(float, line.strip().split(',')))

        # 获取图像帧序号
        frame_number = int(data[0])

        # 存储该帧的目标框信息
        frames_data[frame_number].append(data[2:6])

    # 遍历每一帧，处理目标框
    for frame_number, boxes in tqdm(frames_data.items(), desc="Processing images"):
        filename = f"{frame_number:06d}.jpg"
        file_path = os.path.join(input_dir, filename)

        # 读取图像
        image = cv2.imread(file_path)
        if image is None:
            print(f"Image {filename} not found, skipping...")
            continue

        annotated_image = image.copy()

        # 绘制每个目标框
        for box in boxes:
            x, y, w, h = box
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 1)  # 绿色框，线宽为1

        # 构造输出的文件路径
        output_file_path = os.path.join(output_dir, filename)

        # 保存标注后的图像
        cv2.imwrite(output_file_path, annotated_image)
        print(f"Annotated {filename} saved to {output_file_path}")


# 调用函数
gt_file_path = r'E:\work\small_person_AP\gt_Person_X0.25.txt'  # 替换为gt文件的路径
input_directory = r'E:\work\SR-YOLO\dataset\MOT20-03_481_bicubic_small_X4'  # 替换为源图像文件夹的路径
output_directory = r'E:\work\small_person_AP\gt_small_MOT20-03-481_BICUBIC_X4'  # 替换为要保存标注后图像的路径

draw_bounding_boxes_from_gt(gt_file_path, input_directory, output_directory)