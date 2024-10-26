import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score, auc



def parse_results_file(results_file):
    """
    Parses the results.txt file to extract detections.
    :param results_file: Path to the YOLO results file.
    :return: A dictionary of image detections.
    """
    detections_dict = defaultdict(list)
    current_image_name = None

    current_image_path = None
    detections = []

    with open(results_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Identify lines with image paths (assumed to end with .jpg or .png)
        if line.__contains__('.jpg'):  # Detect image path
            # Update the current image path
            current_image_path = line
            current_image_name = line.split(":")[0].split("/")[-1]
            detections_dict[current_image_name] = []
        else:
            # Parse detection line if it matches the expected format
            if line.__contains__("Enter Image"):
                continue
            if current_image_path:
                # print(line)
                parts = line.split()
                class_name = parts[0].replace(":", "")
                confidence = float(parts[1].replace('%,', ''))
                x_min = int(parts[2])
                y_min = int(parts[3])
                width = int(parts[4])
                height = int(parts[5])
                # detections.append((class_name, confidence, x_min, y_min, width, height))

                detections_dict[current_image_name].append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x_min, y_min, x_min + width, y_min + height]
                })

    return detections_dict


def parse_ground_truths(ground_truths_file):
    """
    Parses the ground truth annotations file.
    :param ground_truths_file: Path to the file containing ground truth annotations.
    :return: A dictionary of ground truth data.
    """
    ground_truths = defaultdict(list)
    with open(ground_truths_file, 'r') as t:
        data = [i.strip() for i in t.readlines()]
    for d in data:
        d = d[:-4] + ".txt"
        with open(d, 'r') as annot_file:
            image_name = d.split("/")[-1][:-4] + ".jpg"
            annot_data = [i.strip() for i in annot_file.readlines()]
            if len(annot_data) == 0:
                ground_truths[image_name] = []
            for line in annot_data:
                if not int(line[0]) in (0, 1):
                    continue
                class_name, x, y, w, h = line.split(" ")
                w = float(w) * 1056
                h = float(h) * 1056
                x_min = 1056 * float(x) - w / 2
                y_min = 1056 * float(y) - h / 2
                x_max = x_min + w
                y_max = y_min + h
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                ground_truths[image_name].append({
                    "class_name": class_names[int(class_name)],
                    'bbox': [x_min, y_min, x_max, y_max]
                })
    return ground_truths


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    :param box1: [x_min, y_min, x_max, y_max]
    :param box2: [x_min, y_min, x_max, y_max]
    :return: IoU value.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


# def calculate_ap_per_class(detections, ground_truths, class_name, iou_threshold=0.5):
#     """
#     Calculates the Average Precision (AP) for a specific class.
#     :param detections: Detections dictionary.
#     :param ground_truths: Ground truths dictionary.
#     :param class_name: The class name to calculate AP for.
#     :param iou_threshold: Minimum IoU to consider a detection as true positive.
#     :return: AP for the class.
#     """
#     y_true = []
#     y_scores = []
#     matched_gt = set()  # Store matched ground truth bounding boxes as tuples
#
#     for image_path in detections:
#         # Get the detections and ground truths for the specific class in the current image
#         image_detections = [d for d in detections[image_path] if d['class_name'] == class_name]
#         # print(image_path, class_name, ground_truths[image_path])
#         image_ground_truths = [gt for gt in ground_truths.get(image_path, []) if gt['class_name'] == class_name]
#
#         for detection in image_detections:
#             best_iou = 0
#             best_gt = None
#             for gt in image_ground_truths:
#                 iou = compute_iou(detection['bbox'], gt['bbox'])
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_gt = tuple(gt['bbox'])  # Convert the ground truth bbox to a tuple for storage
#
#             # Check if the detection is a true positive
#             if best_iou >= iou_threshold and best_gt not in matched_gt:
#                 y_true.append(1)  # True positive
#                 matched_gt.add(best_gt)  # Mark this ground truth as matched
#             else:
#                 y_true.append(0)  # False positive
#
#             y_scores.append(detection['confidence'])
#
#     # Include unmatched ground truth objects as false negatives
#     num_unmatched_gt = len(image_ground_truths) - len(matched_gt)
#     y_true.extend([1] * num_unmatched_gt)
#     y_scores.extend([0] * num_unmatched_gt)
#
#     if len(y_true) == 0:
#         return 0
#
#     # Calculate Average Precision (AP) using sklearn
#     ap = average_precision_score(y_true, y_scores)
#     return ap

def calculate_ap_per_class(detections, ground_truths, class_name, iou_threshold=0.5):
    """
    Calculates the Average Precision (AP) for a specific class.
    :param detections: Detections dictionary.
    :param ground_truths: Ground truths dictionary.
    :param class_name: The class name to calculate AP for.
    :param iou_threshold: Minimum IoU to consider a detection as true positive.
    :return: AP for the class.
    """
    # Lists to store true positives, false positives, and the confidence scores of detections
    tp = []
    fp = []
    y_scores = []

    # Store matched ground truths using a set of tuples representing their bounding boxes
    matched_gts = defaultdict(set)

    # Combine all detections for the class from all images and sort them by confidence
    # Include 'image_path' as part of each detection for traceability
    detections = sorted(
        [
            {**d, 'image_path': img_path}
            for img_path in detections
            for d in detections[img_path]
            if d['class_name'] == class_name
        ],
        key=lambda x: x['confidence'],
        reverse=True
    )

    # Iterate over each detection to determine TP/FP
    for detection in detections:
        image_path = detection['image_path']
        best_iou = 0
        best_gt = None

        # Iterate over ground truths for this image and class
        for gt in ground_truths.get(image_path, []):
            # Use tuple of the bbox as the identifier for matched ground truths
            gt_bbox_tuple = tuple(gt['bbox'])

            # Check if the ground truth has not been matched and calculate IoU
            if gt['class_name'] == class_name and gt_bbox_tuple not in matched_gts[image_path]:
                iou = compute_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_bbox_tuple  # Store the bbox tuple for matching

        # Determine if the detection is a true positive or a false positive
        if best_iou >= iou_threshold and best_gt is not None:
            tp.append(1)  # True positive
            fp.append(0)
            matched_gts[image_path].add(best_gt)  # Mark this ground truth as matched
        else:
            tp.append(0)  # False positive
            fp.append(1)

        # Store the confidence score of the detection
        y_scores.append(detection['confidence'])

    # Calculate cumulative sums of TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    num_gt = sum(len(gts) for gts in ground_truths.values() if any(gt['class_name'] == class_name for gt in gts))

    # Avoid division by zero if there are no ground truth objects for the class
    if num_gt == 0:
        return 0.0

    # Calculate precision and recall
    recall = tp_cumsum / num_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Calculate Average Precision (AP) using the area under the precision-recall curve (AUC)
    ap = auc(recall, precision)
    return ap


def calculate_map(detections, ground_truths, class_names, iou_threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) across all classes.
    :param detections: Detections dictionary.
    :param ground_truths: Ground truths dictionary.
    :param class_names: List of class names.
    :param iou_threshold: IoU threshold for true positive classification.
    :return: mAP score.
    """
    aps = []
    for class_name in class_names:
        ap = calculate_ap_per_class(detections, ground_truths, class_name, iou_threshold)
        print(f"AP for {class_name}: {ap:.4f}")
        aps.append(ap)

    # Calculate mean of all APs to get mAP
    mean_ap = np.mean(aps)
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")
    return mean_ap


def calculate_mean_ap_over_iou_range(detections, ground_truths, class_names, iou_range=(0.5, 0.95, 0.05)):
    """
    Calculates the mean Average Precision (mAP) across a range of IoU thresholds.
    :param detections: Detections dictionary.
    :param ground_truths: Ground truths dictionary.
    :param class_names: List of class names.
    :param iou_range: Tuple with (start, stop, step) for IoU thresholds.
    :return: mAP score across the specified IoU range.
    """
    start, stop, step = iou_range
    iou_thresholds = np.arange(start, stop + step, step)
    aps = []

    for iou_threshold in iou_thresholds:
        print(f"Calculating AP at IoU {iou_threshold:.2f}...")
        aps_per_class = [
            calculate_ap_per_class(detections, ground_truths, class_name, iou_threshold)
            for class_name in class_names
        ]
        print(f"AP at IoU {iou_threshold:.2f}...Person : {aps_per_class[0]:.2f}, CAR : {aps_per_class[1]:.2f}\n")
        mean_ap = np.mean(aps_per_class)
        aps.append(mean_ap)
        print(f"AP at IoU {iou_threshold:.2f}: {mean_ap:.4f}")

    mean_map = np.mean(aps)
    print(f"Mean Average Precision (mAP) across IoU range {start} to {stop}: {mean_map:.4f}")
    return mean_map


# Example usage
results_file = './results_prior_1056_66_middle.txt'  # Path to the YOLOv3 detection results file.
ground_truths_file = 'test.txt'  # Path to the file containing ground truth annotations.
class_names = ['person', 'car']  # Replace with actual class names.

# Parse detections and ground truths
detections = parse_results_file(results_file)
ground_truths = parse_ground_truths(ground_truths_file)

# Calculate mAP
# mean_map = calculate_mean_ap_over_iou_range(detections, ground_truths, class_names, iou_range=(0.5, 0.95, 0.05))

map_score = calculate_map(detections, ground_truths, class_names, iou_threshold=0.5)
