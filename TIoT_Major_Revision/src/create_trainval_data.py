from shutil import copyfile

from sklearn.model_selection import train_test_split
import os


def filter_classes(PATH):
    files = os.listdir(PATH)
    for f in files:
        if not f.__contains__(".txt"):
            continue
        print(f)
        with open(f"{PATH}/{f}", 'r') as s:
            output = []
            data = s.readlines()
            for line in data:
                # line = line.strip()
                if int(line[0]) in (0, 1):
                    output.append(line)
                # else:
                #     print("Other class than car found")
                #     sys.exit(-1)
            with open(f"{PATH}/{f}", 'w') as out_file:
                out_file.writelines(output)


if __name__ == "__main__":
    image_filenames = ["images_2.txt", "images_4.txt", "images_wt.txt"]

    training_images = "train.txt"
    test_images = "test.txt"
    OUTPUT_PATH = "../dataset/trainval"

    agg_names = []
    for filename in image_filenames:
        with open(f"{filename}", 'r') as input_file:
            names = [n.strip() for n in input_file.readlines()]
            agg_names = agg_names + names

    # split data into 90% training and 10% validation
    training_data, test_data = train_test_split(agg_names, test_size=0.1,
                                                random_state=20, shuffle=True)
    with open(training_images, 'w') as t_img:
        for item in training_data:
            print(item)
            src_name = item
            img_name = item.split("/")[-1]
            dst_name = f"{OUTPUT_PATH}/train/{img_name}"
            copyfile(src=src_name[3:], dst=dst_name)
            src_name = f"{src_name[3:-4]}.txt"
            dst_name_gt = f"{dst_name[:-4]}.txt"
            copyfile(src=src_name, dst=dst_name_gt)

            # modify image path relative to darknet directory
            dst_name = f"../{dst_name}"
            t_img.write(f"{dst_name}\n")

    with open(test_images, 'w') as t_img:
        for item in test_data:
            print(item)
            src_name = item
            img_name = item.split("/")[-1]
            dst_name = f"{OUTPUT_PATH}/test/{img_name}"
            copyfile(src=src_name[3:], dst=dst_name)
            src_name = f"{src_name[3:-4]}.txt"
            dst_name_gt = f"{dst_name[:-4]}.txt"
            copyfile(src=src_name, dst=dst_name_gt)

            # modify image path relative to darknet directory
            dst_name = f"../{dst_name}"
            t_img.write(f"{dst_name}\n")

    filter_classes("../dataset/trainval/train")
    filter_classes("../dataset/trainval/test")
