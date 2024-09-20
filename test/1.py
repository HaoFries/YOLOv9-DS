import numpy as np

def calculate_iou(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_precision_recall(gt_boxes, det_boxes, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = len(gt_boxes)

    for det_box in det_boxes:
        matched = False
        for gt_box in gt_boxes:
            iou = calculate_iou(gt_box, det_box[:4])
            if iou >= iou_threshold:
                tp += 1
                fn -= 1
                matched = True
                break
        if not matched:
            fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

def calculate_ap(gt_boxes, det_boxes, iou_threshold=0.5):
    det_boxes = sorted(det_boxes, key=lambda x: x[4], reverse=True)
    precisions = []
    recalls = []

    for i in range(len(det_boxes)):
        precision, recall = calculate_precision_recall(gt_boxes, det_boxes[:i + 1], iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    ap = np.trapz(precisions, recalls)
    return ap

# 示例数据
gt_boxes = [
    (50, 50, 100, 100),
    (150, 150, 100, 100)
]

det_boxes = [
    (60, 60, 100, 100, 0.9),
    (160, 160, 100, 100, 0.8),
    (200, 200, 100, 100, 0.7)
]

precision, recall = calculate_precision_recall(gt_boxes, det_boxes)
ap50 = calculate_ap(gt_boxes, det_boxes)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"AP50: {ap50:.2f}")