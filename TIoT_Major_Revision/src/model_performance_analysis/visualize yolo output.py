import os
import cv2


def parse_yolo_output_file(output_file):
    """
    Parses the YOLO output file to extract image paths and their corresponding detections.

    :param output_file: Path to the output file.
    :return: A list of tuples containing (image_path, detections), where detections is a list of tuples.
    """
    results = []
    with open(output_file, 'r') as f:
        lines = f.readlines()

    current_image_path = None
    detections = []

    for line in lines:
        line = line.strip()
        if line.__contains__('.jpg'):  # Detect image path
            # Save the previous image's detections if there are any
            if current_image_path and detections:
                current_image_path = current_image_path.replace("Enter Image Path: ", "").split(":")[0]
                results.append((current_image_path, detections))
                detections = []  # Reset for the next image

            # Update the current image path
            current_image_path = line
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
                detections.append((class_name, confidence, x_min, y_min, width, height))

    # Add the last image and its detections
    if current_image_path and detections:
        results.append((current_image_path, detections))

    return results


def draw_bounding_boxes(image_path, detections, output_dir):
    """
    Draws bounding boxes on the image and saves the result.

    :param image_path: Path to the input image.
    :param detections: List of detections in the format (class_name, confidence, x_min, y_min, width, height).
    :param output_dir: Directory to save the output image with bounding boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    # Draw each detection on the image
    for detection in detections:
        class_name, confidence, x_min, y_min, width, height = detection
        x_max = x_min + width
        y_max = y_min + height
        color = (0, 255, 0)  # Green for bounding boxes
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Draw the label with the class name and confidence
        label = f"{class_name}: {confidence:.2f}%"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Prepare the output path
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"annotated_{image_name}")

    # Save the annotated image
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")


def visualize_yolo_output(output_file, output_dir):
    """
    Main function to parse the YOLO output file and visualize the detections.

    :param output_file: Path to the YOLO output file.
    :param output_dir: Directory to save the annotated images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parse the output file
    results = parse_yolo_output_file(output_file)

    # Process each image and its detections
    for image_path, detections in results:
        draw_bounding_boxes(image_path, detections, output_dir)


# Example
visualize_yolo_output(
    output_file='./results_prior_128_66_right.txt',  # Replace with the path to the YOLO output file
    output_dir='./prior'  # Replace with the desired output directory
)
