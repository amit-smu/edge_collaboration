"""
takes two images (from different cams) and merges their gt (as if images are merged horizontally)
to create one gt file. This is done for all the test images only.
"""
import cv2
import numpy as np


def visualize_gt(img, annot_file):
    with open(annot_file) as in_file:
        data = in_file.read().split("\n")
        for d in data:
            if len(d) == 0:
                continue
            d = d.split(" ")
            mid_x = float(d[1]) * 1056 * 2
            mid_y = float(d[2]) * 1056
            width = float(d[3]) * 1056 * 2
            height = float(d[4]) * 1056
            # d = [int(float(x) * 1056 * 2) for x in d[1:]]
            # mid_x, mid_y, width, height = d
            x1 = int(mid_x - width / 2)
            y1 = int(mid_y - height / 2)
            x2 = int(mid_x + width / 2)
            y2 = int(mid_y + height / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.resize(img, (1500, 1000))
        cv2.imshow("img", img)
        cv2.waitKey(-1)


if __name__ == "__main__":
    ref_cam = 2
    collab_cam = 3
    img_size = (1056, 1056)

    test_file_path = "./test_frame_numbers.txt"
    annot_dir = "../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_yolo/cam_{}"
    img_dir = "../../../../rpi_hardware/raw_image_processing/data/episode_1/pi_{}_frames_1056_v2"
    output_dir = r"./data/merged_gt/cam_{}_{}".format(ref_cam, collab_cam)

    with open(test_file_path) as test_file:
        framenumbers = test_file.read().split("\n")
        for number in framenumbers:
            print(number)
            f_name_1 = "frame_{}_{}.txt".format(ref_cam, number)
            with open("{}/{}".format(annot_dir.format(ref_cam), f_name_1)) as labels_file_1:
                labels_1 = labels_file_1.read().split("\n")
                img_name = "frame_{}_{}.jpg".format(ref_cam, number)
                img_1 = cv2.imread("{}/{}".format(img_dir.format(ref_cam), img_name))

            f_name_2 = "frame_{}_{}.txt".format(collab_cam, number)
            with open("{}/{}".format(annot_dir.format(collab_cam), f_name_2)) as labels_file_2:
                labels_2 = labels_file_2.read().split("\n")
                img_name = "frame_{}_{}.jpg".format(collab_cam, number)
                img_2 = cv2.imread("{}/{}".format(img_dir.format(collab_cam), img_name))

            # merge gt as if stitching the images horizontally
            output_filename = "frame_{}_{}_{}.txt".format(ref_cam, collab_cam, number)
            with open("{}/{}".format(output_dir, output_filename), 'w') as output_file:
                # adjust gt of first camera
                for label in labels_1:
                    if len(label) == 0:
                        continue
                    label = label.split(" ")
                    new_label = [0, 0, 0, 0, 0]
                    new_label[1] = float(label[1]) / 2
                    new_label[2] = float(label[2])
                    new_label[3] = float(label[3]) / 2
                    new_label[4] = float(label[4])
                    # new_label[1:] = [float(x) / 2 for x in label[1:]]
                    new_label = " ".join([str(l) for l in new_label])
                    output_file.write("{}\n".format(new_label))

                # adjust gt of second camera
                for label in labels_2:
                    if len(label) == 0:
                        continue
                    label = label.split(" ")
                    new_label = [0, 0, 0, 0, 0]
                    new_label[1] = (1 + float(label[1])) / 2
                    new_label[2] = float(label[2])
                    new_label[3] = float(label[3]) / 2
                    new_label[4] = float(label[4])
                    # new_label[1:] = [(1 + float(x)) / 2 for x in label[1:]]
                    new_label = " ".join([str(l) for l in new_label])
                    output_file.write("{}\n".format(new_label))
            # merge images horizontally
            img_merged = np.concatenate((img_1, img_2), axis=1)
            img_merged_name = "frame_{}_{}_{}.jpg".format(ref_cam, collab_cam, number)
            cv2.imwrite("{}/{}".format(output_dir, img_merged_name), img_merged)

            # visualize_gt(img_merged, "{}/{}".format(output_dir, output_filename))
