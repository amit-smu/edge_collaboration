import json

path = "../../rpi_hardware/raw_image_processing/data/episode_1/ep_1_gt_frames_1056.json"
with open(path) as data:
    json_obj = json.loads(data.read())
    print(json_obj)