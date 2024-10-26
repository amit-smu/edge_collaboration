import os
import shutil

if __name__ == "__main__":
    cam = 1
    img_dir = "./data/Images/"
    label_dir = "./data/labels/"
    out_dir = "./test_data/cam_1_4/1056/images/"
    label_out_dir = ""

    for i in range(1400, 2000, 5):
        filename = "C{}_{:08d}.png".format(cam, i)
        print(filename)

        # copy image file
        file_src = "{}/{}".format(img_dir, filename)
        file_dst = "{}/{}".format(out_dir, filename)
        shutil.copyfile(src=file_src, dst=file_dst)

        # copy label file
        label_name = "{}.txt".format(filename[:-4])
        print(label_name)
        label_src = "{}/{}".format(label_dir, label_name)
        label_dst = "{}/{}".format(label_out_dir, label_name)
        shutil.copyfile(src=label_src, dst=label_dst)
        break
