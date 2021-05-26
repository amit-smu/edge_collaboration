WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'

file_path = r"C:\Users\rkamat\Desktop\YOLO\2\yolo-custom-object-detector-master\python\train.txt"

with open(file_path, 'rb') as open_file:
    content = open_file.read()

    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

    with open(file_path, 'wb') as open_file_2:
        open_file_2.write(content)