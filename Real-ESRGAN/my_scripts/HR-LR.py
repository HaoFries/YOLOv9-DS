import cv2
import os


# 缩放图像
def resize_image(input_path, output_path, scale_factors):
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

        # 创建输出文件夹并保存图像
        output_folder = os.path.join(output_path, f'{factor}x')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, os.path.basename(input_path))
        cv2.imwrite(output_file, resized_image)


if __name__ == "__main__":
    input_folder = r"E:\work\SR\Real-ESRGAN-v0.2.9\Real-ESRGAN-v0.2.9\my_scripts\HR"
    output_folder = r"E:\work\SR\Real-ESRGAN-v0.2.9\Real-ESRGAN-v0.2.9\my_scripts\LR"
    scale_factors = [0.25]  # [2, 3, 4]  # 缩放倍数

    # 遍历输入文件夹中的图像文件
    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)
        # 调用函数进行图像缩放
        resize_image(input_path, output_folder, scale_factors)