import random

if __name__ == "__main__":
    # DATASET = "PETS"
    DATASET = "WT"

    # generate trainval data files
    if DATASET == "PETS":
        # trainval
        with open("trainval_70.txt", 'w') as output_file:
            frame_names = []
            for i in range(0, 546):
                # cam 5
                f_name = "frame_5_{:04d}".format(i)
                frame_names.append(f_name)
                # cam 6
                f_name = "frame_6_{:04d}".format(i)
                frame_names.append(f_name)
                # cam 7
                f_name = "frame_7_{:04d}".format(i)
                frame_names.append(f_name)
                # cam 8
                f_name = "frame_8_{:04d}".format(i)
                frame_names.append(f_name)
            # shuffle the list
            random.shuffle(frame_names)

            for name in frame_names:
                output_file.write("{}\n".format(name))
        # test
        with open("test_30.txt", 'w') as output_file:
            frame_names = []
            for i in range(546, 795):
                # cam 5
                f_name = "frame_5_{:04d}".format(i)
                frame_names.append(f_name)
                # cam 6
                f_name = "frame_6_{:04d}".format(i)
                frame_names.append(f_name)
                # cam 7
                f_name = "frame_7_{:04d}".format(i)
                frame_names.append(f_name)
                # cam 8
                f_name = "frame_8_{:04d}".format(i)
                frame_names.append(f_name)
            # shuffle the list
            random.shuffle(frame_names)

            for name in frame_names:
                output_file.write("{}\n".format(name))

    elif DATASET == "WT":
        # trainval
        with open("trainval_70.txt", 'w') as output_file:
            frame_names = []
            for i in range(0, 1405, 5):
                # cam 1
                f_name = "C1_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 2
                f_name = "C2_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 3
                f_name = "C3_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 4
                f_name = "C4_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 5
                f_name = "C5_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 1
                f_name = "C6_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 1
                f_name = "C7_{:08d}".format(i)
                frame_names.append(f_name)
            # shuffle the list
            random.shuffle(frame_names)

            for name in frame_names:
                output_file.write("{}\n".format(name))
        # test
        with open("test_30.txt", 'w') as output_file:
            frame_names = []
            for i in range(1405, 2000, 5):
                # cam 1
                f_name = "C1_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 2
                f_name = "C2_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 3
                f_name = "C3_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 4
                f_name = "C4_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 5
                f_name = "C5_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 1
                f_name = "C6_{:08d}".format(i)
                frame_names.append(f_name)
                # cam 1
                f_name = "C7_{:08d}".format(i)
                frame_names.append(f_name)

            # shuffle the list
            random.shuffle(frame_names)

            for name in frame_names:
                output_file.write("{}\n".format(name))
