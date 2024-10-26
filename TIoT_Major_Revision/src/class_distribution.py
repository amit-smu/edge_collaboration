"""
find class distributions within objects detected by yolo
"""
import os
from copy import deepcopy


def map_obj_class(param):
    pass

def map_index_to_name():
    with open("custom.names", 'r') as coco_data:
        lines = coco_data.readlines()
        names = [l.strip() for l in lines]
        index_to_name = {index : str(n) for index, n in enumerate(names)}
    return index_to_name

if __name__== "__main__":
    class_dist = {}
    dir_list = ["gt_2", "gt_4"]
    dir_path = "./"

    index_to_name = map_index_to_name()


    for dir in dir_list:
        print(f"analysing {dir}\n")

        file_list = os.listdir(f"{dir_path}/{dir}")

        for file in file_list:
            if not file.__contains__(".txt"):
                continue
            print(f"file {file}")
            with open(f"{dir_path}/{dir}/{file}", 'r') as input_file:
                lines = input_file.readlines()
                for l in lines:
                    l = l.strip().split(" ")
                    # l[0] = map_obj_class(l[0])
                    if l[0] in class_dist.keys():
                        class_dist[l[0]] +=1
                    else:
                        class_dist[l[0]] = 1

    # print(index_to_name)
    class_dist_copy = deepcopy(class_dist)
    # print(class_dist_copy)
    class_dist = {index_to_name[int(key)]: value for key, value in class_dist.items()}
    class_dist = sorted(class_dist.items(), key= lambda item:item[1], reverse=True)
    print(class_dist)