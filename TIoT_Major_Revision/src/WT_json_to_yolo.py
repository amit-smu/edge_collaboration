import json
import os


def calculate_icou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    boxA and boxB should be in the format (x_min, y_min, x_max, y_max).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Compute the area of both the bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    iou = inter_area / boxA_area
    return iou


def convert_annotations_for_cropped_image(original_bbox, original_width=1920, original_height=1080,
                                          cropped_size=1056, min_iou=0.5):
    output = None
    # Calculate the cropping offsets
    x_start = (original_width - cropped_size) // 2
    y_start = (original_height - cropped_size) // 2

    # Define the crop area as (x_min, y_min, x_max, y_max)
    crop_area = (x_start, y_start, x_start + cropped_size, y_start + cropped_size)

    x_min, y_min, x_max, y_max = original_bbox
    # Calculate the IoU between the original bounding box and the crop area
    iou = calculate_icou(original_bbox, crop_area)

    # Skip the bounding box if IoU is less than the minimum threshold
    if iou >= min_iou:
        # Adjust the coordinates according to the cropping
        x_min_cropped = x_min - x_start
        y_min_cropped = y_min - y_start
        x_max_cropped = x_max - x_start
        y_max_cropped = y_max - y_start

        # Ensure the bounding box stays within the bounds of the cropped image
        x_min_cropped = max(0, min(cropped_size, x_min_cropped))
        y_min_cropped = max(0, min(cropped_size, y_min_cropped))
        x_max_cropped = max(0, min(cropped_size, x_max_cropped))
        y_max_cropped = max(0, min(cropped_size, y_max_cropped))

        # Filter out any bounding boxes that are completely outside the cropped area
        if x_min_cropped >= cropped_size or y_min_cropped >= cropped_size or x_max_cropped <= 0 or y_max_cropped <= 0:
            return output

        # Reconstruct the adjusted annotation line
        x_min_cropped, y_min_cropped, x_max_cropped, y_max_cropped = convert_to_yolo_format(x_min_cropped,
                                                                                            y_min_cropped,
                                                                                            x_max_cropped,
                                                                                            y_max_cropped)
        adjusted_annotation = f"{0} {x_min_cropped} {y_min_cropped} {x_max_cropped} {y_max_cropped}"
        output = adjusted_annotation
        return output
    else:
        return output


def convert_wildtrack_to_camera_files(input_json_path, output_dir):
    # Load the input JSON file containing all the annotations
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store annotations for each camera
    camera_annotations = {i: [] for i in range(7)}  # Assuming there are 7 cameras with IDs 0-6

    # Iterate over each person annotation
    for person in data:
        person_id = person['personID']
        views = person['views']

        # Process each view for this person
        for view in views:
            view_num = view['viewNum']
            xmin, ymin, xmax, ymax = view['xmin'], view['ymin'], view['xmax'], view['ymax']

            # Ignore invalid bounding boxes where coordinates are -1 (as in the provided data)
            if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1:
                continue

            # Store the annotation in YOLO format
            # Here, class_id is assumed to be 0 for "person"
            class_id = 0

            # transform annotations to 1056x1056
            transformed_annotation = convert_annotations_for_cropped_image([xmin, ymin, xmax, ymax])
            if transformed_annotation is None:
                continue
            # annotation = f"{class_id} {xmin} {ymin} {xmax} {ymax}"
            annotation = transformed_annotation
            # Append to the corresponding camera's list
            camera_annotations[view_num].append(annotation)

    # Write each camera's annotations to a separate text file
    for camera_id, annotations in camera_annotations.items():
        cam_id = int(camera_id) + 1
        output_file = os.path.join(output_dir, f'C{cam_id}_{input_json_path.split("/")[-1][:-5]}.txt')
        with open(output_file, 'w') as f:
            f.write("\n".join(annotations))
        print(f"Saved annotations for Camera {camera_id} to {output_file}")


def convert_to_yolo_format(x_min_cropped, y_min_cropped, x_max_cropped, y_max_cropped):
    x_center = (x_min_cropped + (x_max_cropped - x_min_cropped) / 2) / 1056
    y_center = (y_min_cropped + (y_max_cropped - y_min_cropped) / 2) / 1056
    width = (x_max_cropped - x_min_cropped) / 1056
    height = (y_max_cropped - y_min_cropped) / 1056
    return x_center, y_center, width, height


# Example usage
if __name__ == "__main__":
    INPUT_PATH = "../dataset/wildtrack/annotations_positions/"
    OUTPUT_PATH = "./gt_wt"

    json_list = os.listdir(INPUT_PATH)
    for j in json_list:
        convert_wildtrack_to_camera_files(
            input_json_path=f'../dataset/wildtrack/annotations_positions/{j}',
            # Replace with the path to the input JSON file
            output_dir=OUTPUT_PATH
        )
