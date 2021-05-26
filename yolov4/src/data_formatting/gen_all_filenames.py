output_file = "all_filenames.txt"

with open(output_file, 'w') as output:
    cams = [1, 2, 3, 4, 5, 6, 7]
    for cam in cams:
        for i in range(0, 2000, 5):
            file_name = "C{}_{:08d}.png".format(cam, i)
            output.write("{}\n".format(file_name))
