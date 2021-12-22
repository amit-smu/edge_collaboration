"""
module to extract meta data e.g., #total unique persons, #annotations etc. from the ground truth.
Episode 1 is used for all the computation.
"""
import os
import json

if __name__ == "__main__":
    GT_DIR = r"../../../../rpi_hardware/raw_image_processing/data/episode_1/ground_truth/frame_wise_gt_json/cam_{}"

    person_ids = {1}
    total_annotations = 0
    filenames = os.listdir(GT_DIR.format(2))
    for name in filenames:
        print(name)
        frame_num = name[:-5].split("_")[2]
        for cam in range(1, 4):
            frame_name = "frame_{}_{}.json".format(cam, frame_num)
            with open("{}/{}".format(GT_DIR.format(cam), frame_name)) as in_file:
                obj_list = json.load(in_file)
                total_annotations += len(obj_list)
                for obj in obj_list:
                    obj_id = obj["obj_id"]
                    person_ids.add(obj_id)
            # break
    print("Total Unique Users: {}, Total Annotations: {}\n".format(len(person_ids), total_annotations))
