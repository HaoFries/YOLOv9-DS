import os
import shutil
from tqdm import tqdm

def copy_and_rename_images(input_folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    index = 1
    total_files = sum([len(files) for folder in input_folders for _, _, files in os.walk(folder)])
    with tqdm(total=total_files, desc='Copying images') as pbar:
        for folder in input_folders:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        new_name = f"HR-{index}"
                        file_extension = os.path.splitext(file)[1]
                        new_file_name = f"{new_name}{file_extension}"
                        shutil.copy(os.path.join(root, file), os.path.join(output_folder, new_file_name))
                        index += 1
                        pbar.update(1)

input_folders = [r"E:\MSMT17", r"E:\行人检测\47_CUHK Person Re-identification Datasets\campus",
                 r'E:\数据集下载\Market-1501-v15.09.15', r"E:\DukeMTMC-reID\dukemtmc-reid\DukeMTMC-reID"]  # 输入文件夹路径
output_folder = "E:\work\BoT-SORT-main\Real-ESRGAN\dataset\person-HR"  # 输出文件夹路径

copy_and_rename_images(input_folders, output_folder)