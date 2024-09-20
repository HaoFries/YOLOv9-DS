import os
from PIL import Image

def resize_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理图片文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # 构建输入文件的完整路径
            input_path = os.path.join(input_folder, filename)
            # 打开图片
            with Image.open(input_path) as img:
                # 使用Lanczos插值法放大图片
                new_img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
                # 构建输出文件的完整路径
                output_path = os.path.join(output_folder, filename)
                # 保存新图片
                new_img.save(output_path)
                print(f'Saved resized image to {output_path}')

# 示例使用
input_folder = r'E:\work\SR-YOLO\dataset\MOT20-03_481_bicubic_small_X4'  # 替换为您的输入文件夹路径
output_folder = r'E:\work\small_person_AP\lanczos_small_bicubic_X2'  # 替换为您的输出文件夹路径
resize_images(input_folder, output_folder)