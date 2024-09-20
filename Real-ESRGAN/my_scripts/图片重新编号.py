import os


def rename_images(input_folder):
    # 获取文件夹中所有的图片文件名
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # 从1开始编号，对每个图片文件进行重命名
    for i, image_file in enumerate(image_files, start=1):
        # 构建新的文件名
        new_name = f"HR-{i}.jpg"

        # 旧文件路径
        old_path = os.path.join(input_folder, image_file)
        # 新文件路径
        new_path = os.path.join(input_folder, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed {image_file} to {new_name}")


if __name__ == "__main__":
    input_folder = r"E:\work\SR\Real-ESRGAN-v0.2.9\Real-ESRGAN-v0.2.9\my_scripts\HR"
    rename_images(input_folder)