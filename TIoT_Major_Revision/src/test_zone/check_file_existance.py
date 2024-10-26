import os

if __name__ == "__main__":
    TEST_DIR = "../../dataset/trainval/test"
    INPUT_FILE = "test_remote.txt"

    with open(INPUT_FILE, 'r') as inp_file:
        data = [i.strip() for i in inp_file.readlines()]

        for d in data:
            if not os.path.exists(f"{d}"):
                print(d)
                break
