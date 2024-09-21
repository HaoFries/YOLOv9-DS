from PIL import Image
import os


def resize_images(input_dir, output_dir, scale=4):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 构造完整的文件路径
            file_path = os.path.join(input_dir, filename)

            # 打开图像
            with Image.open(file_path) as img:
                # 计算新的尺寸
                new_width = img.width // scale
                new_height = img.height // scale

                # 使用双三次插值法缩放图像
                resized_img = img.resize((new_width, new_height), Image.BICUBIC)

                # 构造输出的文件路径
                output_file_path = os.path.join(output_dir, filename)

                # 保存缩放后的图像
                resized_img.save(output_file_path)
                print(f'Resized and saved {filename} to {output_file_path}')


# 调用函数
input_directory = ''  # 替换为源图像文件夹的路径
output_directory = ''  # 替换为要保存缩小后的图像的路径
resize_images(input_directory, output_directory)