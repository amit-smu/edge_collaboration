"""script to generate random set of image file names for testing on PETS dataset"""

import random
import os

if __name__ == "__main__":
    N = 500
    output_file_name = "test_{}.txt".format(N)
    dataset_dir = "../dataset/PETS/JPEGImages/"

    images = os.listdir(dataset_dir)
    print("")
    random_numbers = sorted(random.sample(range(0, 796), N))
    print(random_numbers)

    with open(output_file_name, 'w') as output:
        for i in range(0, N):
            img_id = "frame_78_{:04d}".format(random_numbers[i])
            output.write("{}\n".format(img_id))
