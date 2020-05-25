"""
after the crop, if images don't contain persons, then remove them from the
test dataset (evaluator is throwing error for empty files)
"""

import os
import shutil

if __name__ == "__main__":
    dsize = (400, 400)  # desired size (w,h) of image

    # DATASET = "WT"
    DATASET = "PETS"

    print("Dsize : {}".format(dsize))
    print("Dataset : {}".format(DATASET))

    if DATASET == "WT":
        imageset_dir = r"../../dataset/Wildtrack_dataset/PNGImages_cropped_{}x{}".format(dsize[0], dsize[1])
        output_test_dir = "../../dataset/Wildtrack_dataset/ImageSets/Main"

    if DATASET == "PETS":
        imageset_dir = r"../../dataset/PETS_org/JPEGImages_cropped_{}x{}".format(dsize[0], dsize[1])
        output_test_dir = "../../dataset/PETS_org/ImageSets/Main"

    output_test_filename = "test_crop_{}x{}.txt".format(dsize[0], dsize[1])

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