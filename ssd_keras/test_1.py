img_path = "../dataset/PETS/JPEGImages/78_0046.jpg"

t_split = img_path.split("JPEGImages/")
frame_num = img_path[-8:-4]
print("frame_num: {}".format(frame_num))

reference_cam = 7
collaborating_cams = [8, 6, 5]
for c in range(1):
    annot_name = "{}{}_{}.xml".format(collaborating_cams[c], reference_cam, frame_num)
    annot_path = "{}Annotations/{}".format(t_split[0], annot_name)
    print(annot_path)
