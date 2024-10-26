import numpy as np
from collections import defaultdict
from sklearn.metrics import auc

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
