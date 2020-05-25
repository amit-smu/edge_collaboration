from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    annot_dir = "../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/annotations"
    output_dir = "../../dataset/Wildtrack_dataset_full/Wildtrack_dataset/ImageSets/Main"
    train_val_name = "trainval.txt"
    test_name = "test.txt"

    annot_list = os.listdir(annot_dir)
    train_frames, test_frames = train_test_split(annot_list, test_size=0.15)
    print()
    # split training and testing data
    train_frames = sorted(train_frames)
    test_frames = sorted(test_frames)

    # write train, test data to fiels
    with open("{}/{}".format(output_dir, train_val_name), 'w') as train_file:
        for i in train_frames:
            train_file.write("{}\n".format(i.strip(".xml")))

    with open("{}/{}".format(output_dir, test_name), 'w') as test_file:
        for i in test_frames:
            test_file.write("{}\n".format(i.strip(".xml")))
