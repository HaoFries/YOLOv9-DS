import os
from PIL import Image
import shutil
from tqdm import tqdm  # 导入tqdm库

# 获取图片大小的函数
def get_image_size(path):
    with Image.open(path) as img:  # 使用Pillow库中的Image.open函数打开图片文件
        return img.size  # 返回图片的宽度和高度

# 筛选并保存最大的图片
def copy_largest_images(input_folder, output_folder, max_images=30000):
    if not os.path.exists(output_folder):  # 如果输出文件夹不存在，则创建输出文件夹
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图片文件，并按文件大小进行排序
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")]  # 列出输入文件夹中的所有图片文件
    image_files.sort(key=lambda f: os.path.getsize(os.path.join(input_folder, f)), reverse=True)  # 按文件大小进行排序

    with tqdm(total=max_images) as pbar:  # 使用tqdm库创建进度条
        count = 0  # 初始化图片计数器
        for file in image_files:  # 遍历排序后的图片文件列表
            image_path = os.path.join(input_folder, file)  # 获取图片文件的完整路径
            width, height = get_image_size(image_path)  # 调用get_image_size函数获取图片的大小
            new_name = f"HR_{count + 1}.jpg"  # 根据计数器生成新的文件名
            shutil.copy(image_path, os.path.join(output_folder, new_name))  # 复制图片文件到输出文件夹，并按新文件名保存
            count += 1  # 更新图片计数器
            pbar.update(1)  # 更新进度条
            if count >= max_images:  # 如果达到最大图片数量限制，则退出循环
                break

# 输入文件夹路径和输出文件夹路径
input_folder = r"E:\work\SR\Real-ESRGAN\dataset\person-HR"  # 输入文件夹路径
output_folder = r"F:\person-HR_30000"  # 输出文件夹路径

# 调用函数实现筛选并保存最大的图片
copy_largest_images(input_folder, output_folder)