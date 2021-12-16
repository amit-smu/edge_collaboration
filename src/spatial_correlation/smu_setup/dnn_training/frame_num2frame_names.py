"""
create filenames for training/test from frame numbers
"""
import random

if __name__ == "__main__":
    FLAG = "test"
    input_filename = "../../train_test_data_smu_setup/test_frame_numbers.txt"
    output_filename = "../test_1056.txt"
    prefix = "../train_test_data_smu_setup/test_data/single_img_model/Images_1056"

    framenames = []
    with open(input_filename) as input_file:
        frame_numbers = input_file.read().split("\n")
        with open(output_filename, 'w') as out_file:
            for number in frame_numbers:
                if len(number) == 0:
                    continue
                name_1 = "{}/frame_{}_{}.jpg".format(prefix, 1, number)
                framenames.append(name_1)

                if FLAG == "train":
                    name_2 = "{}/frame_{}_{}.jpg".format(prefix, 2, number)
                    name_3 = "{}/frame_{}_{}.jpg".format(prefix, 3, number)
                    framenames.append(name_2)
                    framenames.append(name_3)

            # randomize the names
            random.shuffle(framenames)
            for name in framenames:
                out_file.write("{}\n".format(name))
