"""
module to downsample shared region of images as per the given resolution. This creates a mixed-resolution image.
main file
"""
import cv2
import os
import shutil

if __name__ == "__main__":
    in_data_dir = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/pi_2_frames_1056_v2"
    frame_numbers_path = "./test_frame_numbers.txt"
    out_data_dir = "./data"
    cam_pair = "2_1"
    ref_cam = 2

    resolutions = [1056, 512, 416, 320, 224, 128]
    shared_region_res = 1056

    img_dim = (1056.0, 1056.0)
    spatial_overlap = {
        "2_1": [360, 0, 1010, 1080],
        "2_3": [0, 0, 470, 1080]
    }
    scale = 1056 / 1080
    spatial_overlap[cam_pair] = [int(d * scale) for d in spatial_overlap[cam_pair]]
    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}, Resolution: {}\n".format(cam_pair, spatial_overlap[cam_pair],
                                                                        shared_region_res))
    sh_width = x2 - x1
    sh_height = y2 - y1

    with open(frame_numbers_path) as in_file:
        frame_numbers = in_file.read().split("\n")
    for number in frame_numbers:
        if len(number) == 0:
            continue
        frame_name = "frame_{}_{}.jpg".format(ref_cam, number)
        print(frame_name)
        image = cv2.imread("{}/{}".format(in_data_dir, frame_name))
        assert image is not None

        # extract and downsample the shared region
        shared_region = image[y1:y2, x1:x2]
        sh_reg_new_width = int((sh_width / img_dim[0]) * shared_region_res)
        sh_reg_new_height = int((sh_height / img_dim[1]) * shared_region_res)

        temp = cv2.resize(shared_region, dsize=(sh_reg_new_width, sh_reg_new_height),
                          interpolation=cv2.INTER_AREA)
        shared_region = cv2.resize(temp, dsize=(sh_width, sh_height), interpolation=cv2.INTER_CUBIC)
        image[y1:y2, x1:x2] = shared_region
        # save the resulting image
        cv2.imwrite("{}/{}".format(out_data_dir, frame_name), image)
