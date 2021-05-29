import os
import shutil

if __name__ == "__main__":
    in_dir = r"G:\Datasets\Wildtrack_dataset\PNGImages"
    label_dir = r"D:\My Drive\yolov3\single_img_model\data\labels"
    out_dir = r"D:\My Drive\yolov3\test_data_1920\data_org\Images"

    test_file = r"D:\My Drive\yolov3\test_data_1920\test_7_5.txt"
    with open(test_file) as test_names:
        test_names = test_names.read().split("\n")
        for name in test_names:
            src_name = "{}/{}".format(in_dir, name)
            dst_name = "{}/{}".format(out_dir, name)
            print(src_name)
            shutil.copyfile(src=src_name, dst=dst_name)

            # copy label
            src_label = "{}/{}.txt".format(label_dir, name[:-4])
            dst_label = "{}.txt".format(dst_name[:-4])
            shutil.copyfile(src=src_label, dst=dst_label)
            print(src_label)
            # break
