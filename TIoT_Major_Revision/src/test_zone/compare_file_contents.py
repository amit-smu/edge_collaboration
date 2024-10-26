FILE_1 = "images_2.txt"
FILE_2 = "images_2_remote.txt"

with open(FILE_1, 'r') as file_1:
    data_1 = [i.strip() for i in file_1.readlines()]
    data_1 = sorted(data_1)
with open(FILE_2, 'r') as file_2:
    data_2 = [i.strip() for i in file_2.readlines()]
    data_2 = sorted(data_2)

print(list(set(data_1) - set(data_2)))
