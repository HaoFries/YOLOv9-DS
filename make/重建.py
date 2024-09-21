import os
import shutil

def rename_and_save_images(source_folder, destination_folder):
    # 获取源文件夹中所有的子文件夹
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

    # 遍历每个子文件夹
    for folder in subfolders:
        # 获取子文件夹中所有的图片文件
        image_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.jpg')]

        # 遍历每个图片文件
        for i, image_file in enumerate(image_files):
            # 生成新的文件名
            new_file_name = f"{i + 1}.jpg"
            # 拼接新的文件路径
            new_file_path = os.path.join(destination_folder, new_file_name)
            # 复制文件到目标文件夹并重命名
            shutil.copyfile(image_file, new_file_path)

    print("图片重命名并保存完成")

# 指定源文件夹和目标文件夹
source_folder = r"E:\work\bicubic-plusplus-main\dataset\MOT20_train"
destination_folder = r"E:\work\bicubic-plusplus-main\dataset\train_LR_20"

# 调用函数
rename_and_save_images(source_folder, destination_folder)