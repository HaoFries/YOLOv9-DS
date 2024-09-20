import os
import numpy as np
from tqdm import tqdm


def parse_gt_file(gt_file):
    gt_data = {}
    with open(gt_file, 'r') as file:
        for line in file:
            frame_id, _, x, y, w, h, flag, category, confidence = map(float, line.strip().split(','))
            if flag == 1 and category == 1:
                if frame_id not in gt_data:
                    gt_data[int(frame_id)] = []
                gt_data[int(frame_id)].append((x, y, w, h, confidence))
    return gt_data

def parse_detection_folder(det_folder):
    det_data = {}
    for det_file in os.listdir(det_folder):
        # 取文件名的前六位作为帧ID，并转换为整数
        frame_id = int(det_file[:6])
        detections = []
        with open(os.path.join(det_folder, det_file), 'r') as file:
            for line in file:
                _, x, y, w, h, confidence = map(float, line.strip().split())
                detections.append((x, y, w, h, confidence))
        det_data[frame_id] = detections
    return det_data


def calculate_iou(gt_box, det_box):
    x1_min, y1_min, w1, h1 = gt_box[:4]
    x2_min, y2_min, w2, h2 = det_box[:4]

    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    gt_area = w1 * h1
    det_area = w2 * h2
    iou = inter_area / float(gt_area + det_area - inter_area)
    return iou


def calculate_precision_recall(gt_data, det_data, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = sum(len(boxes) for boxes in gt_data.values())

    for frame_id, detections in det_data.items():
        if frame_id not in gt_data:
            fp += len(detections)
            continue

        gt_boxes = gt_data[frame_id]
        matched_gt = set()

        for det_box in detections:
            matched = False
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue

                iou = calculate_iou(gt_box, det_box)
                if iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(i)
                    matched = True
                    break
            if not matched:
                fp += 1

        fn -= len(matched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall


def calculate_ap(gt_data, det_data, iou_threshold=0.5):
    all_detections = []
    for frame_id, boxes in det_data.items():
        all_detections.extend([(frame_id, box) for box in boxes])

    all_detections = sorted(all_detections, key=lambda x: x[1][4], reverse=True)

    precisions = []
    recalls = []

    # 使用 tqdm 包装计算 AP 的循环
    for i in tqdm(range(1, len(all_detections) + 1), desc="Calculating AP"):
        subset_detections = {}
        # 这里的 j 应该从 0 开始，以包含第一个检测结果
        for j in range(i):
            frame_id, box = all_detections[j]
            if frame_id not in subset_detections:
                subset_detections[frame_id] = []
            subset_detections[frame_id].append(box)

        precision, recall = calculate_precision_recall(gt_data, subset_detections, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    ap = np.trapz(precisions, recalls)
    return ap


# 示例使用
gt_file = r'E:\work\small_person_AP\gt_Person_X0.25.txt'  # GT文件路径
det_folder = r'E:\work\small_person_AP\yolov8_small_person_labels\yolov8s_yuan'  # 检测结果文件夹路径

gt_data = parse_gt_file(gt_file)
det_data = parse_detection_folder(det_folder)

ap50 = calculate_ap(gt_data, det_data)
print(f"AP50: {ap50:.4f}")