#======================Market-1501文件夹图片下采样三倍、模糊、失真和添加噪声====================
from PIL import Image, ImageFilter
import numpy as np
import os

input_folder = r'E:\work\SR-YOLO\Functional_module_example\dataset_make\input_image\market-1501\gt_bbox'
output_folder = r'E:\work\SR-YOLO\Functional_module_example\dataset_make\output_image\market-1501_blurred+noisy'
resize_factor = 3

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 模糊处理
def apply_blur(image):
    blurred_image = image.filter(ImageFilter.BLUR)
    return blurred_image


# 失真处理
def apply_distortion(image):
    distorted_image = image.resize((image.width // resize_factor, image.height // resize_factor), Image.ANTIALIAS)
    distorted_image = distorted_image.resize((image.width, image.height), Image.ANTIALIAS)
    return distorted_image


# 添加噪声
def apply_noise(image):
    image_array = np.array(image)
    noise = np.random.normal(0, 20, image_array.shape)
    noisy_image = Image.fromarray(np.uint8(np.clip(image_array + noise, 0, 255)))
    return noisy_image


for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(os.path.join(input_folder, filename))

        # # 缩小三倍
        # distorted_image = img.resize((img.width // resize_factor, img.height // resize_factor), Image.ANTIALIAS)

        # # 下采样缩小图片像素值
        # img = img.resize((img.width // 3, img.height // 3))

        # 模糊处理
        blurred_img = apply_blur(img)
        blurred_img.save(os.path.join(output_folder, 'blurred_' + filename))

        # # 失真处理
        # distorted_img = apply_distortion(img)
        # distorted_img.save(os.path.join(output_folder, 'distorted_' + filename))

        # 添加噪声
        noisy_img = apply_noise(img)
        noisy_img.save(os.path.join(output_folder, 'noisy_' + filename))

        #模糊处理+添加噪声
        blurred_noisy_img = apply_noise(blurred_img)
        blurred_noisy_img.save(os.path.join(output_folder, 'blurred+noisy_' + filename))

