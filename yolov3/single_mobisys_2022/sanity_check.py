import os
import sys


def get_key(x):
    x = x[:-4].split("_")[-1]
    return int(x)


if __name__ == "__main__":
    pi_1_dir = r"../../rpi_hardware/raw_image_processing/data/episode_1/pi_1_frames"
    pi_2_dir = r"../../rpi_hardware/raw_image_processing/data/episode_1/pi_2_frames"
    pi_3_dir = r"../../rpi_hardware/raw_image_processing/data/episode_1/pi_3_frames"

    pi_1_files = sorted(os.listdir(pi_1_dir), key=lambda x: get_key(x))
    pi_2_files = sorted(os.listdir(pi_2_dir), key=lambda x: get_key(x))
    pi_3_files = sorted(os.listdir(pi_3_dir), key=lambda x: get_key(x))

    for index, file_2 in enumerate(pi_3_files):
        print(index, file_2)
        file_1 = pi_1_files[index]
        file_2 = pi_2_files[index]
        file_3 = pi_3_files[index]

        if get_key(file_1) != get_key(file_2) or get_key(file_1) != get_key(file_3):
            print(file_1, file_2, file_3)
            sys.exit(-1)
