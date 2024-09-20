###########################废弃的程序，不再使用#################
import os
from tqdm import tqdm
import time

def calculate_iou(gt_box, det_box):
    # 提取坐标
    x1_min, y1_min, w1, h1 = gt_box
    x2_min, y2_min, w2, h2 = det_box

    # 计算矩形的边界
    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    # 计算交集的坐标
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 计算交集面积
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 计算两个矩形的面积
    gt_area = w1 * h1
    det_area = w2 * h2

    # 计算IoU
    iou = inter_area / float(gt_area + det_area - inter_area)

    return iou

def parse_gt_file(gt_file):
    gt_data = []
    with open(gt_file, 'r') as file:
        for line in file:
            frame_id, _, x, y, w, h, flag, category, visibility = map(float, line.strip().split(','))
            if flag == 1 and category == 1:
                gt_data.append((int(frame_id), (x, y, w, h)))
    return gt_data

def parse_detection_file(det_file):
    det_data = []
    with open(det_file, 'r') as file:
        for line in file:
            _, x, y, w, h, confidence = map(float, line.strip().split())
            det_data.append((x, y, w, h))
    return det_data

def count_total_small_objects(detection_dir, width_threshold=50, height_threshold=50):
    total_small_objects = 0
    for filename in tqdm(os.listdir(detection_dir), desc="Counting small objects"):
        filepath = os.path.join(detection_dir, filename)
        det_data = parse_detection_file(filepath)
        for _, _, w, h in det_data:
            if w < width_threshold and h < height_threshold:
                total_small_objects += 1
    return total_small_objects

def match_detections(gt_data, detection_dir, width_threshold=50, height_threshold=50):
    true_positives = 0
    total_gt = len(gt_data)

    iou_list = []

    for frame_id, gt_box in tqdm(gt_data, desc="Processing frames"):
        # 生成文件名
        filename = os.path.join(detection_dir, f"{frame_id:06d}.txt")

        # 读取检测文件
        if os.path.exists(filename):
            det_data = parse_detection_file(filename)

            max_iou = 0  # 初始化最大 IoU

            for det_box in det_data:
                # 计算 IoU
                iou = calculate_iou(gt_box, det_box[:4])

                # 保留最大的 IoU 值
                if iou > max_iou:
                    max_iou = iou

            iou_list.append(max_iou)

            # 判断最大 IoU 是否达到阈值
            if max_iou >= 0.5:
                true_positives += 1

    # 计算精确度和召回率
    total_small_objects = count_total_small_objects(detection_dir, width_threshold, height_threshold)
    precision = true_positives / total_small_objects if total_small_objects else 0  # 应当改为det中的小目标是否在gt文件中存在
    recall = true_positives / total_gt if total_gt else 0  # 思路正确，从gt文件中找det文件中是否有，gt目标是否检测全了。
    print(true_positives)
    print(total_gt)

    return precision, recall, iou_list

def calculate_ap(iou_list, threshold=0.5):  # 需要加入置信度。
    # 计算 AP 值
    iou_list.sort(reverse=True)
    true_positives = sum(iou >= threshold for iou in iou_list)
    precision = true_positives / len(iou_list) if iou_list else 0
    recall = true_positives / len(iou_list) if iou_list else 0
    return precision * recall

# 主程序
gt_file = r'E:\work\SR-YOLO\dataset\samll_MOT_0.2\MOT20-03\gt\gt_small_scale_pedestrians.txt'  # GT文件路径
detection_dir = r'E:\work\YOLO\yolov9-main\MOT20-03-481-X2\yolov9_e-converted_1280_detect_MOT20-03_481_0.01_0.45\labels'  # 检测结果文件夹路径

gt_data = parse_gt_file(gt_file)
start_time = time.time()
precision, recall, iou_list = match_detections(gt_data, detection_dir)
end_time = time.time()
average_precision = calculate_ap(iou_list)

elapsed_time = end_time - start_time
print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"平均精度 (AP): {average_precision:.2f}")
print(f"总耗时: {elapsed_time:.2f} 秒")