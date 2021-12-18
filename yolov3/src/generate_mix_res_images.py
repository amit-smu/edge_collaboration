"""
module to downsample shared region of images as per the given resolution
"""
import cv2
import os
import shutil

if __name__ == "__main__":
    cam_pair = "5_7"

    in_data_dir = r"./test_data_1920/data_org/Images"
    img_dim = (1920.0, 1080.0)

    resolutions = [1056, 512, 416, 320, 224, 128]
    spatial_overlap = {
        "1_4": [6, 4, 908, 1074],
        "5_7": [51, 139, 1507, 1041]
    }

    x1, y1, x2, y2 = spatial_overlap[cam_pair]
    print("Cam pair: {}, Spatial Overlap : {}\n".format(cam_pair, spatial_overlap[cam_pair]))
    sh_width = x2 - x1
    sh_height = y2 - y1

    fnames = os.listdir(in_data_dir)
    for res in resolutions:
        print("Resolution: {}\n".format(res))

        out_data_dir = r"./test_data_1920/single_img_model/cam_{}/data_{}/Images/".format(cam_pair, res)
        # print(out_data_dir)

        for name in fnames:
            if name.__contains__("txt"):
                continue
            if name.__contains__("C{}".format(cam_pair.split("_")[0])):
                print("image name :{}\n".format(name))
                image = cv2.imread("{}/{}".format(in_data_dir, name))
                # assert image.shape == (1056, 1056, 3)

                # extract and downsample the shared region
                shared_region = image[y1:y2, x1:x2]
                sh_reg_new_width = int((sh_width / img_dim[0]) * res)
                sh_reg_new_height = int((sh_height / img_dim[1]) * res)

                temp = cv2.resize(shared_region, dsize=(sh_reg_new_width, sh_reg_new_height),
                                  interpolation=cv2.INTER_AREA)
                shared_region = cv2.resize(temp, dsize=(sh_width, sh_height), interpolation=cv2.INTER_CUBIC)
                image[y1:y2, x1:x2] = shared_region

                # save the resulting image
                cv2.imwrite("{}/{}".format(out_data_dir, name), image)

                # copy the label as well
                src_label = "{}/{}.txt".format(in_data_dir, name[:-4])
                dst_label = "{}/{}.txt".format(out_data_dir, name[:-4])
                print(src_label)
                print(dst_label)
                shutil.copyfile(src=src_label, dst=dst_label)
                # break
        # break
