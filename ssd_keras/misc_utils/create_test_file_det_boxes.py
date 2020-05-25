"""
create test image file from all the images in a specified directory
"""

import os
import shutil

if __name__ == "__main__":

    # DATASET = "WT"
    DATASET = "PETS"
    ref_cam = 7
    collab_cam = 8

    print("Dataset : {}".format(DATASET))

    if DATASET == "WT":
        imageset_dir = r"../../dataset/Wildtrack_dataset/PNGImages_cropped_r{}_c{}_{}".format(ref_cam, collab_cam,
                                                                                              ref_cam)
        output_test_dir = "../../dataset/Wildtrack_dataset/ImageSets/Main"

    if DATASET == "PETS":
        imageset_dir = r"../../dataset/PETS_org/JPEGImages_cropped_r{}_c{}_{}".format(ref_cam, collab_cam,
                                                                                      ref_cam)
        output_test_dir = "../../dataset/PETS_org/ImageSets/Main"

    output_test_filename = "test_crop_r{}_c{}.txt".format(ref_cam, collab_cam)

    # temp_image_names = os.listdir(imageset_dir)

    test_image_names = []

    image_names = os.listdir(imageset_dir)
    for img_name in image_names:
        img_name = img_name[:-4]
        test_image_names.append(img_name)
    test_image_names = sorted(test_image_names)
    print("Total images in test: {}\n".format(len(test_image_names)))

    # write to new test image file
    with open("{}/{}".format(output_test_dir, output_test_filename), 'w') as output_file:
        for img_name in test_image_names:
            output_file.write("{}\n".format(img_name))
