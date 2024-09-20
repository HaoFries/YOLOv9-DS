import cv2
import os
import pathlib
from skimage.metrics import structural_similarity as ssim

# 定义原始 HR 图像文件夹和重建 HR 图像文件夹路径
hr_original_folder = r'E:\work\SR\Real-ESRGAN-v0.2.9\Real-ESRGAN-v0.2.9\my_scripts\bicubic_NOISE_down\HR'
hr_reconstructed_folder = r'E:\work\SR\Real-ESRGAN-v0.2.9\Real-ESRGAN-v0.2.9\my_scripts\eval-SR\X4plus_person_HR_first\210000pth\SR-X2  '

# 初始化用于存储 PSNR 和 SSIM 的列表
psnr_values = []
ssim_values = []

# 遍历原始 HR 图像文件夹中的图像
for filename in os.listdir(hr_original_folder):
    if filename.startswith('HR-'):
        original_path = os.path.join(hr_original_folder, filename)
        reconstructed_filename = filename.replace('HR-', 'HR-').replace('.jpg', '_X2.jpg')
        reconstructed_path = os.path.join(hr_reconstructed_folder, reconstructed_filename)

        # 加载原始 HR 图像和重建 HR 图像
        hr_original = cv2.imread(original_path)
        hr_reconstructed = cv2.imread(reconstructed_path)

        # 转换为灰度图像
        hr_original_gray = cv2.cvtColor(hr_original, cv2.COLOR_BGR2GRAY)
        hr_reconstructed_gray = cv2.cvtColor(hr_reconstructed, cv2.COLOR_BGR2GRAY)

        # 确保原始图像和重建图像具有相同的尺寸
        height = min(hr_original_gray.shape[0], hr_reconstructed_gray.shape[0])  # 取高度的最小值
        width = min(hr_original_gray.shape[1], hr_reconstructed_gray.shape[1])  # 取宽度的最小值

        hr_original_gray = hr_original_gray[:height, :width]  # 裁剪原始图像
        hr_reconstructed_gray = hr_reconstructed_gray[:height, :width]  # 裁剪重建图像

        # 计算 PSNR
        psnr = cv2.PSNR(hr_original_gray, hr_reconstructed_gray)
        psnr_values.append(psnr)

        # 计算 SSIM
        ssim_value = ssim(hr_original_gray, hr_reconstructed_gray)
        ssim_values.append(ssim_value)

        # 输出每张图片的 PSNR 和 SSIM
        print(f'Image: {filename}')
        print(f'PSNR: {psnr}')
        print(f'SSIM: {ssim_value}')
        print('-------------------------')

# 计算平均 PSNR 和 SSIM
average_psnr = sum(psnr_values) / len(psnr_values)
average_ssim = sum(ssim_values) / len(ssim_values)

print(f'Average PSNR: {average_psnr}')
print(f'Average SSIM: {average_ssim}')

# 创建保存结果的文件路径
folder_names = pathlib.Path(hr_reconstructed_folder).parts[-2:]
txt_filename = "_".join(folder_names) + ".txt"
txt_filepath = os.path.join(hr_reconstructed_folder, txt_filename)

# 打开并写入结果到TXT文件
with open(txt_filepath, "w") as file:
    file.write("Average PSNR: {:.14f}\n".format(average_psnr))
    file.write("Average SSIM: {:.15f}\n\n".format(average_ssim))

    for i, filename in enumerate(os.listdir(hr_original_folder)):
        if filename.startswith('HR-'):
            file.write(f'Image: {filename}\n')
            file.write(f'PSNR: {psnr_values[i]}\n')
            file.write(f'SSIM: {ssim_values[i]:.14f}\n')
            file.write('-------------------------\n')

print(f"Results saved to: {txt_filepath}")