############将所有的图片和标签都整理到YOLO格式的文件夹并改为YOLO格式的标签#############
import os
from pathlib import Path
import shutil

def process_and_save_images_and_labels(image_dirs, gt_dirs, output_image_dir, output_label_dir):
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    current_index = 1  # 序号从1开始

    for image_dir, gt_dir in zip(image_dirs, gt_dirs):
        image_dir = Path(image_dir)
        gt_dir = Path(gt_dir)

        images = sorted(image_dir.glob("*.jpg"))
        for image in images:
            # 获取图片文件名的数字部分并去掉前导零
            image_number = int(image.stem)  # 获取图片文件名的序号部分，并转换为整数
            gt_file = gt_dir / f"{image_number}.txt"  # 使用整数形式的文件名进行匹配

            # 复制并重命名图片
            output_image_path = output_image_dir / f"{current_index:06d}.jpg"
            shutil.copy(image, output_image_path)

            # 复制并重命名gt文件
            output_label_path = output_label_dir / f"{current_index:06d}.txt"
            if gt_file.exists():
                shutil.copy(gt_file, output_label_path)
            else:
                print(f"Warning: GT file {gt_file} not found for image {image}")

            current_index += 1

    print(f"Processing complete. Images saved to {output_image_dir}, Labels saved to {output_label_dir}.")

# 定义输入和输出路径
image_dirs = [
    r"E:\work\SR-YOLO\dataset\MOT20\train\MOT20-01\img1",
    r"E:\work\SR-YOLO\dataset\MOT20\train\MOT20-02\img1",
    r"E:\work\SR-YOLO\dataset\MOT20\train\MOT20-03\img1",
    r"E:\work\SR-YOLO\dataset\MOT20\train\MOT20-05\img1"
]

gt_dirs = [
    r"E:\work\MOT20_dataset_make\gt_file-with-all-yolo-labels-not-integrated\gt_01",
    r"E:\work\MOT20_dataset_make\gt_file-with-all-yolo-labels-not-integrated\gt_02",
    r"E:\work\MOT20_dataset_make\gt_file-with-all-yolo-labels-not-integrated\gt_03",
    r"E:\work\MOT20_dataset_make\gt_file-with-all-yolo-labels-not-integrated\gt_05"
]

output_image_dir = r"E:\work\YOLO\yolov9-main\datasets\images"
output_label_dir = r"E:\work\YOLO\yolov9-main\datasets\labels"

# 处理图片和标签文件
process_and_save_images_and_labels(image_dirs, gt_dirs, output_image_dir, output_label_dir)