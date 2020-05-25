"""
module to convert the PETS dataset used in regression craeteion, to a format usable by ssd_eval_XX_det_boxes files.
"""

import os
import cv2
import shutil

if __name__ == "__main__":
    input_img_dir = r"../../dataset/PETS_1/JPEGImages/"
    input_annot_dir = r"../../dataset/PETS_1/Annotations/"

    ref_cam = 7
    collab_cam = 8

    output_img_dir = r"../../dataset/PETS_1/JPEGImages_det_boxes_r{}_c{}_{}/".format(ref_cam, collab_cam, collab_cam)
    output_annot_dir = r"../../dataset/PETS_1/Annotations_det_boxes_r{}_c{}_{}/".format(ref_cam, collab_cam,
                                                                                        collab_cam)

    if os.path.exists(output_img_dir):
        shutil.rmtree(output_img_dir)
    if os.path.exists(output_annot_dir):
        shutil.rmtree(output_annot_dir)
    os.mkdir(output_img_dir)
    os.mkdir(output_annot_dir)

    img_name_list = os.listdir(input_img_dir)
    for name in img_name_list:
        if name.startswith("frame_{}{}".format(collab_cam, ref_cam)):
            print(name)
            frame_num = name.split("_")[2][:-4]
            output_img_name = "frame_{}_{}.jpg".format(collab_cam, frame_num)
            output_annot_name = "frame_{}_{}.xml".format(collab_cam, frame_num)
            shutil.copyfile(src="{}/{}".format(input_img_dir, name),
                            dst="{}/{}".format(output_img_dir, output_img_name))

            shutil.copyfile(src="{}/{}.xml".format(input_annot_dir, name[:-4]),
                            dst="{}/{}".format(output_annot_dir, output_annot_name))
            # break
        # elif name.startswith("frame_{}{}".format(collab_cam, ref_cam)):
        #     print(name)
        #     frame_num = name.split("_")[2][:-4]
        #     output_img_name = "frame_{}_{}.jpg".format(collab_cam, frame_num)
        #     output_annot_name = "frame_{}_{}.xml".format(collab_cam, frame_num)
        #     shutil.copyfile(src="{}/{}".format(input_img_dir, name),
        #                     dst="{}/{}".format(output_img_dir, output_img_name))
        #
        #     shutil.copyfile(src="{}/{}.xml".format(input_annot_dir, name[:-4]),
        #                     dst="{}/{}".format(output_annot_dir, output_annot_name))
