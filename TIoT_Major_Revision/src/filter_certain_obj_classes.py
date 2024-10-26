import os
import sys

if __name__ == "__main__":
    PATH = "../dataset/trainval/test"
    files = os.listdir(PATH)

    for f in files:
        if not f.__contains__(".txt"):
            continue
        print(f)
        with open(f"{PATH}/{f}", 'r') as s:
            output = []
            data = s.readlines()
            for line in data:
                # line = line.strip()
                if int(line[0]) in (0,1):
                    output.append(line)
                # else:
                #     print("Other class than car found")
                #     sys.exit(-1)
            with open(f"{PATH}/{f}", 'w') as out_file:
                out_file.writelines(output)
