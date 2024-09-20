import os
from tqdm import tqdm
import time

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

def parse_detection_file(det_file):
    det_data = []
    with open(det_file, 'r') as file:
        for line in file:
            _, x, y, w, h, confidence = map(float, line.strip().split())
            det_data.append((x, y, w, h, confidence))
    return det_data

def match_detections(gt_data, detection_dir):
    true_positives = 0
    total_detections = 0
    total_gt_boxes = 0  # 用于存储gt框的总数

    for filename in tqdm(os.listdir(detection_dir), desc="Matching detections"):
        frame_id = int(filename[:6])
        if frame_id in gt_data:
            total_gt_boxes += len(gt_data[frame_id])  # 增加gt框的数量
            filepath = os.path.join(detection_dir, filename)
            det_data = parse_detection_file(filepath)
            for det_box in det_data:
                total_detections += 1
                max_iou = 0
                for gt_box in gt_data[frame_id]:
                    iou = calculate_iou(gt_box, det_box[:4])
                    max_iou = max(max_iou, iou)
                if max_iou >= 0.5:
                    true_positives += 1

    precision = true_positives / total_detections if total_detections else 0
    recall = true_positives / total_gt_boxes if total_gt_boxes else 0

    return precision, recall 

gt_file = r'E:\work\small_person_AP\gt_Person_X0.25.txt'
detection_dir = r'E:\work\small_person_AP\yolov8_small_person_labels\yuan_X2'

start_time = time.time()
gt_data = parse_gt_file(gt_file)
precision, recall = match_detections(gt_data, detection_dir)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Total time: {elapsed_time:.2f} seconds")