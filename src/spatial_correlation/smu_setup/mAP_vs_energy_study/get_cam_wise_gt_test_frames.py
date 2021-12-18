"""
copy yolo gt for all cams (only for test frame numbers)
"""
import shutil

if __name__ == "__main__":
    input_filename = "./test_frame_numbers.txt"
    annot_dir = "../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_yolo/cam_{}"
    output_dir = "./data/cam_wise_gt_yolo/cam_{}"
    with open(input_filename) as in_file:
        framenumbers = in_file.read().split("\n")
        for number in framenumbers:
            if len(number) == 0:
                continue
            print(number)
            for cam in range(1, 4):
                frame_name = "frame_{}_{}.txt".format(cam, number)
                src_dir = annot_dir.format(cam)
                dst_dir = output_dir.format(cam)
                shutil.copyfile("{}/{}".format(src_dir, frame_name), "{}/{}".format(dst_dir, frame_name))
