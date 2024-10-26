def load_name_to_index_mapper():
    with open("custom.names", 'r') as names:
        names_list = names.readlines()
        names_list = [n.strip() for n in names_list]
        names_mapper = {str(n): index for index, n in enumerate(names_list)}
        return names_mapper


def process_result(result):
    image_size = (1056, 1056)
    processed_results = []
    name_to_index_mapper = load_name_to_index_mapper()
    for r in result:
        obj_name, obj_coord = r.split(":")
        # process object name
        if obj_name in name_to_index_mapper.keys():
            obj_name = name_to_index_mapper[obj_name]
        else:
            continue

        # process object coordinates
        obj_coord = obj_coord.split("%, ")[1]
        # convert object coordinates to normalized format as per yolo
        x, y, w, h = obj_coord.split(" ")
        x = (int(x) + int(w) / 2) / 1056
        y = (int(y) + int(h) / 2) / 1056
        w = int(w) / 1056
        h = int(h) / 1056
        # obj_coord = [int(i)/1056 for i in obj_coord.split(" ")]
        obj_coord = f"{x} {y} {w} {h}"
        processed_results.append(f"{obj_name} {obj_coord}")
    return processed_results


def split_yolo_output(output_file, images_file, output_dir):
    with open(output_file, 'r') as f:
        lines = f.readlines()

    with open(images_file, 'r') as f:
        images = [line.strip() for line in f]

    image_results = {}
    current_image = None

    for line in lines:
        if any(image in line for image in images):
            current_image = line.strip().split('/')[-1].split(' ')[0]  # Extract image name
            image_results[current_image] = []
        elif current_image is not None and line.strip():
            image_results[current_image].append(line.strip())

    # Save each image's results to a separate file
    for image, results in image_results.items():
        output_path = f"{output_dir}/{image[:-5]}.txt"
        with open(output_path, 'w') as out_file:
            results = process_result(results)
            out_file.write("\n".join(results))

    print(f"Results have been split into individual files in {output_dir}")


# Call the function
split_yolo_output('results_4.txt', 'images_4.txt', 'gt_4')
