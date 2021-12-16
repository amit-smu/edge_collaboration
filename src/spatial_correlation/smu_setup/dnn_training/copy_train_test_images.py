"""
moduel to copy training/test images to respective directory based on teh set of train/test frames
"""
import os
import shutil

if __name__ == "__main__":
    cam_dir = r"../../train_test_data_smu_setup/data_org/pi_{}_frames_1056_v2"

    numbers_file = r"../../train_test_data_smu_setup/test_frame_numbers.txt"
    dst_dir = r"../../train_test_data_smu_setup/test_data/single_img_model/Images/"

    with open(numbers_file) as in_file:
        numbers = in_file.read().split("\n")
        for n in numbers:
            if len(n) == 0:
                continue
            for cam in range(1, 4):
                current_cam_dir = cam_dir.format(cam)
                frame_name = "frame_{}_{}".format(cam, n)
                shutil.copyfile("{}/{}.jpg".format(current_cam_dir, frame_name),
                                dst="{}/{}.jpg".format(dst_dir, frame_name))
                shutil.copyfile("{}/{}.txt".format(current_cam_dir, frame_name),
                                dst="{}/{}.txt".format(dst_dir, frame_name))

            # break
