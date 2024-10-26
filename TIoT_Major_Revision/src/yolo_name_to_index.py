import os


def load_class_names(names_file):
    with open(names_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def convert_names_to_indices(result_file, names_file, output_file):
    class_names = load_class_names(names_file)
    name_to_index = {name: str(i) for i, name in enumerate(class_names)}

    with open(result_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            for name, index in name_to_index.items():
                line = line.replace(name, index)
            f.write(line)

# Usage

# convert_names_to_indices('result.txt', 'cfg/obj.names', 'result_indices.txt')

if __name__ == "__main__":
    INPUT_PATH = "./gt_15"
    filenames = os.listdir(INPUT_PATH)
    for name in filenames:
        if name.__contains__(".txt"):
            convert_names_to_indices(result_file=f"{INPUT_PATH}/{name}",
                                     names_file="coco.names",
                                     output_file=f"{INPUT_PATH}/{name}")