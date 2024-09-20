import cv2
import os
import numpy as np
import time
import math
import torch
import threading
import argparse
import traceback
import torchvision
from PIL import Image
# import torch.nn.functional as F
# from model import Generator
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
import sys

def filter_rectangles(rectangle_list, diff_image):
    # start_time = time.time()  # 获取函数开始执行的时间

    diff_np = np.array(diff_image)  # 将diff_image转换为NumPy数组

    filtered_rectangles_list = []  # 创建一个空列表，用于存储过滤后的矩形
    for rect in rectangle_list:
        # 获取矩形的坐标和尺寸信息
        x, y, w, h = rect

        # 将矩形区域从diff_np中提取出来
        subarray = diff_np[y:y+h, x:x+w]

        # 判断矩形是否被包含在差异图像中
        if np.any(subarray == 1):
            filtered_rectangles_list.append(rect)

    # end_time = time.time()  # 获取函数执行结束的时间
    # print("函数运行时间：", end_time - start_time, "秒")  # 显示函数的运行时间

    return filtered_rectangles_list
def process_rectangle_and_save_images(image, rectangle_list, output_folder, image_number):
    """
    将矩形绘制在原始图像上，并保存带有矩形的图像。

    Args:
    image: 输入图像的对象。
    rectangle_list (list): 包含矩形位置信息的列表，格式为 (x, y, 宽度, 高度)。
    output_folder (str): 保存带有矩形的图像的文件夹路径。
    image_number (int): 图像编号，用于保存文件时的命名。
    """
    # 创建输入图像的副本
    image_copy = image.copy()

    # 绘制矩形
    for rect in rectangle_list:
        x, y, w, h = rect
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 在图像副本上绘制矩形

    # 保存带有矩形的图像
    output_image_path = os.path.join(output_folder, "rectangles_on_image_{}.jpg".format(image_number))
    cv2.imwrite(output_image_path, image_copy)
    return image_copy
# 查看矩形的图片效果
def process_samll_rectangles(rectangles_list, width, height, area_threshold=36, distance_threshold=50):
    """
    对矩形列表进行处理，移除面积小于阈值且与其他矩形不重叠的矩形.

    Args:
    rectangles_list (list): 包含矩形信息的列表，每个矩形由左上角坐标 (x, y) 和宽度 w、高度 h 组成
    area_threshold (int): 面积阈值，小于该阈值的矩形将被移除
    distance_threshold (int): 距离阈值，用于判断两个矩形是否不重叠
    width (int): 矩形所在区域的宽度
    height (int): 矩形所在区域的高度

    Returns:
    list: 处理后的矩形列表，移除面积小于阈值且与其他矩形不重叠的矩形

    """
    result = list(rectangles_list)
    to_remove = set()

    for i in range(len(result)):
        x, y, w, h = result[i]
        if w * h < area_threshold:
            center_x = x + w // 2
            center_y = y + h // 2
            x1 = max(0, center_x - w // 2 - distance_threshold)
            x2 = min(width, center_x + w // 2 + distance_threshold)
            y1 = max(0, center_y - h // 2 - 2 * distance_threshold)
            y2 = min(height, center_y + h // 2 + 2 * distance_threshold)

            for j in range(len(result)):
                if j != i:
                    x_, y_, w_, h_ = result[j]
                    if not (y1 >= y_ + h_ or y2 <= y_ or x1 >= x_ + w_ or x2 <= x_):
                        # 当前处理的矩形和其他矩形有重叠
                        to_remove.add(i)
                        break

    result = [rect for i, rect in enumerate(result) if i not in to_remove]
    return result
# 处理小且偏远的矩形
def is_overlap(rect1, rect2):
    # 检查两个矩形在x轴和y轴上是否有重叠
    x_overlap = (rect1[0] < rect2[0] + rect2[2]) and (rect1[0] + rect1[2] > rect2[0])
    y_overlap = (rect1[1] < rect2[1] + rect2[3]) and (rect1[1] + rect1[3] > rect2[1])
    return x_overlap and y_overlap
def group_overlapping_rectangles(rectangles):
    # 将重叠的矩形分组
    groups = []
    for rect in rectangles:
        new_group = [rect]
        for group in groups:
            if any(is_overlap(rect, r) for r in group):
                new_group.extend(group)
                groups.remove(group)
        groups.append(new_group)
    return groups
def merge_overlapping_rectangles(rectangles):
    # 合并重叠的矩形
    groups = group_overlapping_rectangles(rectangles)
    result = []
    for group in groups:
        # 计算合并后矩形的位置和大小
        min_x = min(group, key=lambda rect: rect[0])[0]
        min_y = min(group, key=lambda rect: rect[1])[1]
        max_x = max(group, key=lambda rect: rect[0]+rect[2])[0] + max(group, key=lambda rect: rect[0]+rect[2])[2]
        max_y = max(group, key=lambda rect: rect[1]+rect[3])[1] + max(group, key=lambda rect: rect[1]+rect[3])[3]
        result.append((min_x, min_y, max_x - min_x, max_y - min_y))
    return result
# 重叠矩形合并
def is_contained(rect1, rect2):
    # 判断矩形 rect1 是否被矩形 rect2 包含的函数
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2
def remove_contained_rectangles(rectangles):
    # 从矩形列表中移除被其他矩形包含的矩形的函数
    result = []
    for i in range(len(rectangles)):
        is_contained_by_other = False
        for j in range(len(rectangles)):
            if i != j and is_contained(rectangles[i], rectangles[j]):
                is_contained_by_other = True
                break
        if not is_contained_by_other:
            result.append(rectangles[i])
    return result
# 筛除完全重叠矩形
def process_small_rectangles(rectangles_list, image_width, image_height, area_threshold, max_area):
    """
    对小于指定面积的矩形进行处理，将其放大到不大于指定面积的面积。

    Args:
    rectangles_list (list): 包含矩形位置信息的列表，格式为 (x, y, 宽度, 高度)。
    area_threshold (int): 面积阈值，用于判断矩形的面积。
    max_area (int): 最大面积限制。
    image_width (int): 图像的宽度。
    image_height (int): 图像的高度。

    Returns:
    list: 经过处理后的矩形位置信息的列表，格式为 (x, y, 宽度, 高度)。
    """
    magnify_rectangles_list = []
    for rect in rectangles_list:
        x, y, w, h = rect
        area = w * h
        if area < area_threshold:
            n = math.sqrt(max_area / (w * h))
            new_w = n * w
            new_h = n * h
            new_x = max(0, x + (w - new_w) / 2)
            new_y = max(0, y + (h - new_h) / 2)
            new_x1 = int(max(0, new_x))
            new_x2 = int(min(image_width, new_x + new_w))
            new_y1 = int(max(0, new_y))
            new_y2 = int(min(image_height, new_y + new_h))
            new_width = new_x2 - new_x1
            new_height = new_y2 - new_y1
            magnify_rectangles_list.append((new_x1, new_y1, new_width, new_height))
        else:
            new_x1 = int(max(0, x))
            new_x2 = int(min(image_width, x + w))
            new_y1 = int(max(0, y))
            new_y2 = int(min(image_height, y + h))
            new_width = new_x2 - new_x1
            new_height = new_y2 - new_y1
            magnify_rectangles_list.append((new_x1, new_y1, new_width, new_height))

    return magnify_rectangles_list
# 对小于指定面积的矩形进行处理，将其放大到不大于指定面积的面积。
def rect_redivision(rect_list):  #图形分裂再组合
    #重叠图形消除
    rect_list.sort(key=lambda x: (x[0], x[1]))
    temp_list1=[]
    while len(rect_list) > 0:
        temp_list1 = temp_list1 + rect_list[:1]
        x0, y0, w0, h0 = temp_list1[-1]
        temp_list2 = rect_list[1:]
        for k in range(0, len(temp_list2)):
            x1, y1, w1, h1 = temp_list2[k]
            if x1 >= x0 + w0 or y1 >= y0 + h0 or y1 + h1 <= y0:
                continue
            else:
                temp_list2[k] = (0, 0, 0, 0)
                temp_list2.append((x1, y1, w1, max(0, y0 - y1)))
                temp_list2.append((x1, y0 + h0, w1, max(0, y1 + h1 - y0 - h0)))
                temp_list2.append((x0 + w0, y1, max(0, x1 + w1 - x0 - w0), h1))
        rect_list = Mloop(temp_list2)
        rect_list.sort(key=lambda x: (x[0], x[1]))
    rect_list=temp_list1
    #相临的X（或Y）相同的图形消除
    temp_list = []
    while rect_list!=temp_list:
        temp_list = rect_list
        elimSampleXY(rect_list, 0)
        elimSampleXY(rect_list, 1)
    return rect_list
def Mloop(rect_list):###消除无效的（w,h为0）图形
    for k in range(len(rect_list) - 1, -1, -1):
        if rect_list[k][2] == 0 or rect_list[k][3] == 0:
            del rect_list[k]
    return rect_list
def elimSampleXY(rect_list,i):##合并X（或Y）相同且相临的图形
    if i==1:Coexposition(rect_list)
    rect_list.sort(key=lambda x: (x[0], x[2]))
    temp_list1 = []
    while len(rect_list) > 0:
        temp_list1 = temp_list1 + rect_list[:1]
        x0, y0, w0, h0 = temp_list1[-1]
        temp_list2 = rect_list[1:]
        for k in range(0, len(temp_list2)):
            x1, y1, w1, h1 = temp_list2[k]
            if x1 > x0:
                break
            if y0 + h0 == y1:
                temp_list1[-1] = (x0, y0, w0, h0 + h1)
                temp_list2[k] = (x0 + w0, y1, w1 - w0, h1)
                break
            else:
                continue
        rect_list = Mloop(temp_list2)
        rect_list.sort(key=lambda x: (x[0], x[2]))
    rect_list = temp_list1
    if i == 1: Coexposition(rect_list)
    return rect_list
def Coexposition(rect_list):###坐标交换位置
    temp_list = []
    for k in range(0, len(rect_list)):
        x0, y0, w0, h0 = rect_list[k]
        temp_list.append((y0, h0, x0, w0))
    rect_list = temp_list
    return rect_list
def cut_image_rectangles_list(image, sr_rectangles_list):
    """
    对输入的图像根据给定的矩形列表进行剪切，并返回剪切后的图像列表。

    参数:
    image: numpy数组
        输入的图像，可以是OpenCV格式的图像。
    sr_rectangles_list: list
        包含矩形坐标的列表，每个矩形表示为一个元组 (x, y, w, h)

    返回值:
    list
        剪切后的图像列表，每个元素表示为numpy数组。
    """
    cropped_images = []
    for rect in sr_rectangles_list:
        x, y, w, h = rect
        cropped_image = image[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
    return cropped_images
def enhance_image_np(img, model_path, model_name, device, outscale=2):
    """
    根据提供的模型路径和设备，选择相应的模型并增强输入的OpenCV图像。

    参数:
    img: numpy数组
        OpenCV格式的输入图像。
    model_path: str
        模型文件的路径。
    device: torch.device
        指定运行增强操作的设备。

    返回:
    numpy数组
        增强后的图像。
    """
    # 从模型路径中提取模型名称
    model_name = os.path.basename(model_name).split('.')[0]

    # 根据模型名称选择相应的模型
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
    else:
        raise ValueError("Unsupported model name extracted from model path: {}".format(model_name))

    # 初始化Restorer
    upsampler = RealESRGANer(
        scale=netscale,  # 在网络中使用的上采样比例因子通常为2或4。
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,  # 10
        pre_pad=0,
        half=True,  # fp32
        gpu_id=device.index if device.type == "cuda" else None
    )

    # 增强图片
    output, _ = upsampler.enhance(img, outscale=outscale)

    return output
def process_image(img, model_path, model_name, device, enhanced_images, idx):
    try:
        enhanced_image = enhance_image_np(img, model_path, model_name, device, outscale=2)
        enhanced_images[idx] = enhanced_image
    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        traceback.print_exc()
def process_images(image, sr_rectangles_list, model_path, model_name, device):
    image_copy = np.copy(image)
    cropped_images = cut_image_rectangles_list(image_copy, sr_rectangles_list)

    enhanced_images = [None] * len(cropped_images)  # 初始化一个与裁剪图像列表相同长度的列表

    threads = []

    for idx, img in enumerate(cropped_images):
        thread = threading.Thread(target=process_image, args=(img, model_path, model_name, device, enhanced_images, idx))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return enhanced_images
def overlay_images(original_image, processed_rectangles_list, enhanced_images):
    """
    将增强后的图像覆盖到原始图像的指定位置。

    参数:
    original_image: np.ndarray
        原始大图像数组。
    processed_rectangles_list: list
        处理后的矩形信息元组列表。
    enhanced_images: list
        增强后的图像列表，与处理后的矩形信息元组列表一一对应。

    返回值:
    np.ndarray
        包含增强后图像的原始大图像数组。
    """
    overlaid_image = np.copy(original_image)

    for rect, enhanced_img in zip(processed_rectangles_list, enhanced_images):
        if enhanced_img is not None:  # 确保 enhanced_img 不是 None
            x, y, w, h = rect
            overlaid_image[y:y + h, x:x + w] = enhanced_img
            # overlaid_image[y:y + h, x:x + w] = enhanced_img.astype(np.uint8)  # 将增强后的图像转换为无符号8位整数类型

    return overlaid_image

def lanczos_enlarge_image(image, scale_factor):
    """
    对输入的图像进行指定倍数的放大，使用Lanczos插值法。

    参数:
    image: numpy数组
        输入的图像，可以是OpenCV格式的图像。
    scale_factor: int, optional
        放大倍数

    返回值:
    numpy数组
        放大后的图像。
    """
    # 创建输入图像的副本
    input_image = np.copy(image)

    # 将图像的宽和高都放大指定倍数，使用Lanczos插值法
    enlarged_image = cv2.resize(input_image, (input_image.shape[1]*scale_factor, input_image.shape[0]*scale_factor), interpolation=cv2.INTER_LANCZOS4)

    # 返回放大后的图像
    return enlarged_image
def process_scaled_rectangles(sr_rectangles_list, scale_factor=4):
    """
    将给定矩形列表中的每个矩形信息元组中的数字都乘以指定放大倍数，并返回处理后的矩形信息元组列表。

    参数:
    sr_rectangles_list: list
        包含矩形信息元组的列表，每个元组表示为 (x, y, w, h)
    scale_factor: int, optional
        放大倍数，默认为4

    返回值:
    list
        处理后的矩形信息元组列表
    """
    processed_rectangles = []
    for rect in sr_rectangles_list:
        x, y, w, h = rect
        processed_rect = (x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor)
        processed_rectangles.append(processed_rect)

    return processed_rectangles
def save_image_to_folder(image, folder_path, index):
    """
    保存图像到指定文件夹中。

    Args:
        image (numpy.ndarray): 要保存的图像数组。
        folder_path (str): 要保存图像的文件夹路径。
        index (int): 图像的索引，用于命名保存的图像文件。
    """

    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 将索引转换为六位数字编号，并在前面补零
    index_str = f"{index:06}"

    # 为图片命名
    image_name = f"{index_str}.jpg"  # 将保存的图像格式改为JPG
    image_path = os.path.join(folder_path, image_name)

    # 保存图像为JPG格式
    cv2.imwrite(image_path, image)
def rgb_matrix(original_image, final_matrix, image_number, output_folder):
    """
    将给定的二维矩阵中数值为1的位置在RGB图像中标记为绿色，并保存修改后的图像到指定文件夹中。

    参数:
    original_image (numpy.ndarray): 原始RGB图像的NumPy数组表示。
    final_matrix (numpy.ndarray): 二维矩阵，标记需要修改颜色的位置为1。
    image_number (int): 图像编号。
    output_folder (str): 输出文件夹路径。

    返回:
    PIL.Image.Image: 修改后的RGB图像对象。
    """
    # 将NumPy数组转换为PIL图像对象
    original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    # 创建原始图像的副本
    image_copy = original_image_pil.copy()

    # 读取RGB图片并将其转换为三个二维矩阵（R、G、B）
    r_matrix, g_matrix, b_matrix = image_copy.split()

    # 将R、G、B矩阵转换为NumPy数组
    r_array = np.array(r_matrix)
    g_array = np.array(g_matrix)
    b_array = np.array(b_matrix)

    # 将二维矩阵中数值为1的位置对应的R、G、B矩阵中的数值进行修改
    r_array[final_matrix == 1] = 0
    g_array[final_matrix == 1] = 255
    b_array[final_matrix == 1] = 0

    # 将修改后的R、G、B矩阵合并为RGB图片
    modified_image = Image.merge('RGB', (Image.fromarray(r_array), Image.fromarray(g_array), Image.fromarray(b_array)))

    # 构建完整的文件路径
    output_file_path = os.path.join(output_folder, '{:06d}.jpg'.format(image_number))

    # 保存图像到文件夹中
    modified_image.save(output_file_path)

    return modified_image
def main():
    parser = argparse.ArgumentParser(description='描述你的程序功能')

    # 添加命令行参数
    parser.add_argument('--input', type=str, default='',
                        help='输入图像或文件夹的路径')
    parser.add_argument('--output_outline_rectangle', type=str,
                        default='',
                        help='检测边缘轮廓的矩形框的保存路径')
    parser.add_argument('--area_threshold', type=int, default=900, help='矩形框的面积阈值')
    parser.add_argument('--output_rgb_matrix', type=str, default='', help='')

    # 添加更多参数...

    args = parser.parse_args()
    output_redivision = r''  # sr前矩形框
    output_SPD_Real_ESRGAN = r''  # SR后YOLO前图片

    image_number = 1
    gray_images = []
    diff_images_4 = []
    diff_images_7 = []
    final_matrix = None  # 设置final_matrix的初始值为None
    # 创建输出窗口
    # cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)
    start_time = time.time()  # 记录整个循环开始时间
    while True:
        frame_start_time = time.time()  # 记录每帧开始时间  args.input

        image_path = os.path.join(args.input, '{:06d}.jpg'.format(image_number))
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
        else:
            image = None

        # 读取图像文件
        if image is None:
            break  # 停止循环

        # 生成灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 生成灰度图和轮廓矩形列表
        # 计算图像的宽度和高度
        height, width = image.shape[:2]

        # 创建原始图片的副本
        input_image_copy = image.copy()

        # 使用 Canny 边缘检测
        edges = cv2.Canny(gray, 100, 200)

        # 寻找边缘轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建矩形列表
        outline_rectangle_list = []
        max_area = 250  # 最大面积
        aspect_ratio_threshold = 1.5  # 宽高比阈值
        # 处理轮廓
        for contour in contours:
            area = cv2.contourArea(contour)

            # 过滤掉面积太大的物体
            if area > max_area:
                continue

            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算宽高比
            aspect_ratio = float(w) / h if h != 0 else 0

            # 过滤掉宽高比太大的物体
            if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1 / aspect_ratio_threshold:
                continue
            cv2.rectangle(input_image_copy, (x, y), (x + w, y + h), (0, 255, 255), 1)
            outline_rectangle_list.append((x, y, w, h))  # outline_rectangle_list: 包含矩形框坐标和尺寸的元组列表。


        # 将带有矩形框的图像保存到指定路径
        output_outline_path = os.path.join(args.output_outline_rectangle, 'outline_{}.jpg'.format(image_number))  # image_number: 用于输出文件名的编号。
        cv2.imwrite(output_outline_path, input_image_copy)
        gray_images.append(gray)  # 灰度图放入列表1
        if len(gray_images) == 2:  # 如果列表中已经有2个image
            diff = cv2.absdiff(gray_images[0], gray_images[1])
            threshold = 25  # 可设定的阈值，控制差异程度
            _, thresholded = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)  # 将阈值化后的图像转换为0和1的二值矩阵
            diff_image = thresholded  # 两帧差分 01 矩阵
            diff_images_7.append(diff_image)  # 差分 01 矩阵,放入列表
            if len(diff_images_7) == 7:
                for i in range(0, 7, 2):
                    diff_images_4.append(diff_images_7[i])  # 将diff_images_8中的第2、4、6、8个差分图添加到diff_images_4
                diff_images_7.pop(0)  # 移除第一张
                # 确保输入列表中包含4个矩阵
                if len(diff_images_4) != 4:
                    raise ValueError("Input list must contain 4 matrices")

                # 将4个矩阵重叠在一起
                temp_matrix = np.add(diff_images_4[0], diff_images_4[1])
                temp_matrix = np.add(temp_matrix, diff_images_4[2])
                overlapped_matrix = np.add(temp_matrix, diff_images_4[3])

                # 记录重叠次数为2到4的位置
                overlap_2_to_4_indices = np.where((overlapped_matrix >= 2) & (overlapped_matrix <= 4))

                # 将记录的位置对应到最后一个矩阵中，将这些位置的数值改为0
                diff_images_4[3][overlap_2_to_4_indices] = 0
                final_matrix = diff_images_4[3]

                rgb_matrix(image, final_matrix, image_number, args.output_rgb_matrix)  # 观察效果，对应到RGB图片上

                diff_images_4.clear()  # 清空diff_images_4

            if final_matrix is not None:
                diff_np = np.array(final_matrix)  # 将diff_image转换为NumPy数组

                filtered_rectangles_list = []  # 创建一个空列表，用于存储过滤后的矩形
                for rect in outline_rectangle_list:
                    # 获取矩形的坐标和尺寸信息
                    x, y, w, h = rect

                    # 将矩形区域从diff_np中提取出来
                    subarray = diff_np[y:y + h, x:x + w]

                    # 判断矩形是否被包含在差异图像中
                    if np.any(subarray == 1):
                        filtered_rectangles_list.append(rect)
                # 从第8帧开始4diff叠加筛除非人反光物体
            else:
                # 第2张图片的轮廓矩形和第1和第2张图片的差分区域的交集筛选，输出矩形，耗时的函数0.1s-0.2s
                diff_np = np.array(diff_image)  # 将diff_image转换为NumPy数组

                filtered_rectangles_list = []  # 创建一个空列表，用于存储过滤后的矩形
                for rect in outline_rectangle_list:
                    # 获取矩形的坐标和尺寸信息
                    x, y, w, h = rect

                    # 将矩形区域从diff_np中提取出来
                    subarray = diff_np[y:y + h, x:x + w]

                    # 判断矩形是否被包含在差异图像中
                    if np.any(subarray == 1):
                        filtered_rectangles_list.append(rect)


            merge_rectangles_list = merge_overlapping_rectangles(filtered_rectangles_list)  # 重叠矩形合并
            remove_contained_rectangles_list = remove_contained_rectangles(merge_rectangles_list)  # 筛除完全重叠矩形
            enlarge_rectangles_list = process_small_rectangles(remove_contained_rectangles_list, width, height,
                                                               area_threshold=360, max_area=360)  # 矩形放大到行人大小
            sr_rectangles_list = rect_redivision(enlarge_rectangles_list)  # 矩形重分割

            bicubic_x2_images = bicubic_enlarge_image(image, scale_factor=2)  # 原始图片使用bicubic_X2

            # 定义设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 显示设备信息
            if device.type == 'cuda':
                print('使用 GPU 设备：', torch.cuda.get_device_name(0))
            else:
                print('使用 CPU 设备')
            model_name = 'RealESRGAN_x4plus'  # 定义模型格式
            model_path = r'E:\work\SR-YOLO\Fast_pretreatment_Real_ESRGAN\weights\net_g_210000.pth'  # 使用的模型
            enhanced_images = process_images(image, sr_rectangles_list, model_path, model_name, device)  # 批量real-esrgan处理图像
            processed_rectangles = process_scaled_rectangles(sr_rectangles_list, scale_factor=2)  # 坐标对应X2倍

            # real-esrgan_X2图像覆盖到bicubic_X2图像
            final_image = overlay_images(bicubic_x2_images, processed_rectangles, enhanced_images)

            # 保存bicubic+real-esrgan_X2图片到文件夹
            save_image_to_folder(final_image, output_SPD_Real_ESRGAN, image_number)

            gray_images.pop(0)  # 更新列表
            # # 显示当前帧图片
            # cv2.imshow("Processed Frame", image_output)
            #
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出循环
            #     break

        end_time = time.time()  # 记录每帧结束时间
        frame_time = end_time - frame_start_time  # 计算每帧运行时间
        print(f"第{image_number}帧使用的时间：{frame_time}秒")  # 在终端显示每帧的运行时间
        image_number += 1
    cv2.destroyAllWindows()
    total_end_time = time.time()  # 记录整个循环结束时间
    total_time = total_end_time - start_time  # 计算整个循环的总运行时间
    print(f"整个循环运行完成所需时间：{total_time}秒")  # 在终端显示整个循环的总运行时间

if __name__ == "__main__":#这些代码(main函数)只会在当前模块作为主程序运行时执行，而在被其他模块导入时不会执行
    main()