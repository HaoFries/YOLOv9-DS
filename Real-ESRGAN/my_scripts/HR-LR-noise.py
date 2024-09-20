import cv2
import numpy as np
import os


# 添加高斯噪声
def add_noise(image, noise_level):
    h, w, c = image.shape
    noise = np.random.normal(0, noise_level, (h, w, c))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


# 缩放图像并添加噪声
def resize_image(input_path, output_path, scale_factors, noise_level):
    image = cv2.imread(input_path)

    # 获取原始文件名
    filename = os.path.basename(input_path)
    # 将原图片名称中的数字提取出来
    num = ''.join(filter(str.isdigit, filename))
    # 构建新的文件名
    new_filename = f"LR-{num}.jpg"

    for factor in scale_factors:
        # 使用双三次插值法缩放图像
        resized_image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        # 添加噪声
        noisy_image = add_noise(resized_image, noise_level)

        # 创建输出文件夹并保存图像
        output_folder = os.path.join(output_path, f'{factor}x')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, new_filename)
        cv2.imwrite(output_file, noisy_image)

if __name__ == "__main__":
    input_folder = r"E:\work\SR\Real-ESRGAN-v0.2.9\Real-ESRGAN-v0.2.9\my_scripts\HR"
    output_folder = r"E:\work\SR\Real-ESRGAN-v0.2.9\Real-ESRGAN-v0.2.9\my_scripts\LR"
    scale_factors = [0.5] # [0.5  0.33  0.25]  # 缩放倍数
    noise_level = 20  # 噪声水平

    # 遍历输入文件夹中的图像文件
    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)
        # 调用函数进行图像缩放和添加噪声
        resize_image(input_path, output_folder, scale_factors, noise_level)