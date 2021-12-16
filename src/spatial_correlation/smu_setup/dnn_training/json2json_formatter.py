"""
module to convert a single json file to multiple files (each for each frame in each camera)
"""
import os
import json


def split_combined_json_file():
    in_filename = "labels_by_vithurson_ep_1.json"
    with open("{}/{}".format(IN_DIR_1, in_filename)) as in_file:
        combined_json_obj = json.load(in_file)
        keys = sorted(combined_json_obj.keys(), key=lambda x: int(x))

        for key in keys:
            frame_json = combined_json_obj[key]
            json_all_cams = [frame_json["cam_1"], frame_json["cam_2"], frame_json["cam_3"]]

            # put json in multiple files
            for cam in range(1, 4):
                out_filename = "frame_{}_{}.json".format(cam, key)
                with open("{}/cam_{}/{}".format(OUT_DIR, cam, out_filename), 'w') as out_file:
                    json.dump(json_all_cams[cam - 1], out_file, indent=4)
                    print()


def split_multi_json_to_frame_wise_json(in_dir):
    filenames = os.listdir(in_dir)
    filenames = sorted(filenames, key=lambda x: int(x[:-5].split("_")[1]))
    for name in filenames:
        with open("{}/{}".format(in_dir, name)) as in_file:
            json_obj = json.load(in_file)
            key = name[:-5].split("_")[1]
            json_obj = json_obj[key]
            json_all_cams = [json_obj["cam_1"], json_obj["cam_2"], json_obj["cam_3"]]
            for cam in range(1, 4):
                out_filename = "frame_{}_{}.json".format(cam, key)
                with open("{}/cam_{}/{}".format(OUT_DIR, cam, out_filename), 'w') as out_file:
                    json.dump(json_all_cams[cam - 1], out_file, indent=4)


if __name__ == "__main__":
    IN_DIR_1 = r"./data/episode_1/ground_truth/"
    IN_DIR_2 = r"./data/episode_1/ground_truth/gt_json"
    OUT_DIR = r"./data/episode_1/ground_truth/frame_wise_gt_json"

    # split_combined_json_file()
    split_multi_json_to_frame_wise_json(IN_DIR_2)
