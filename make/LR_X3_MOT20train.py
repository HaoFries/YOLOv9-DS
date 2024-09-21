#======================文件夹MOT20_num_group图片下采样三倍专用====================
import os
from PIL import Image

# 设置原始图片文件夹和处理后的图片文件夹
input_folder = r'E:\work\SR-YOLO\Functional_module_example\dataset\output_image\MOT20_num_group'
output_folder = r'E:\work\SR-YOLO\Functional_module_example\dataset\output_image\downsamplingX3\MOT20_train'

# 创建处理后的图片文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历MOT20_num_group中的文件夹
for foldername in os.listdir(input_folder):
    if foldername.startswith('person_'):
        # 创建处理后的person_{x}文件夹
        output_subfolder = os.path.join(output_folder, foldername)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # 遍历person_{x}文件夹中的图片
        for filename in os.listdir(os.path.join(input_folder, foldername)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 打开图片
                img = Image.open(os.path.join(input_folder, foldername, filename))

                # 下采样缩小图片像素值
                img = img.resize((img.width // 3, img.height // 3))

                # 保存处理后的图片
                img.save(os.path.join(output_subfolder, filename))
