import os
import shutil

if __name__ == "__main__":
    INPUT_DIR = ""
    OUT_DIR = ""
    file_names = os.listdir(INPUT_DIR)

    for f_name in file_names:
        src_img_path = ""
        shutil.copyfile(src=src_img_path, dst=dst_img_path)
