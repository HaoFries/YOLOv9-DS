import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


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


def parse_detection_file(det_file_path):
    frame_id = int(os.path.basename(det_file_path)[:6])
    detections = []
    with open(det_file_path, 'r') as file:
        for line in file:
            _, x, y, w, h, confidence = map(float, line.strip().split())
            detections.append((x, y, w, h, confidence))
    return (frame_id, detections)


def parse_detection_folder_parallel(det_folder):
    det_files = [os.path.join(det_folder, f) for f in os.listdir(det_folder)]
    with Pool() as pool:
        results = pool.map(parse_detection_file, det_files)
    return dict(results)


def calculate_precision_recall_fast(gt_data, det_data_sorted, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = sum(len(boxes) for boxes in gt_data.values())
    current_frame_id = -1
    gt_boxes = []

    for frame_id, det_box in det_data_sorted:
        if frame_id != current_frame_id:
            current_frame_id = frame_id
            gt_boxes = gt_data.get(frame_id, [])
            matched_gt = set()

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

def calculate_ap(gt_data, det_data_sorted, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = sum(len(boxes) for boxes in gt_data.values())

    precisions = []
    recalls = []

    for index, (frame_id, det_box) in enumerate(det_data_sorted, start=1):
        if frame_id in gt_data:
            matched = False
            for i, gt_box in enumerate(gt_data[frame_id]):
                iou = calculate_iou(gt_box, det_box)
                if iou >= iou_threshold:
                    if not matched:
                        matched = True
                        tp += 1
                        gt_data[frame_id].pop(i)  # Remove matched ground truth to prevent double matching
                        break
            if not matched:
                fp += 1
        else:
            fp += 1  # Detection without corresponding ground truth

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    # Calculate the actual AP value
    ap = np.trapz(sorted(precisions), sorted(recalls))
    return ap

# 示例使用
gt_file = r'E:\work\small_person_AP\gt_Person_X0.25.txt'
det_folder = r'E:\work\small_person_AP\yolov8_small_person_labels\yuan_X2'

gt_data = parse_gt_file(gt_file)
det_data = parse_detection_folder_parallel(det_folder)
det_data_sorted = sorted([(frame_id, box) for frame_id, boxes in det_data.items() for box in boxes],
                         key=lambda x: x[1][4], reverse=True)

ap50 = calculate_ap(gt_data, det_data_sorted)
print(f"AP50: {ap50:.4f}")