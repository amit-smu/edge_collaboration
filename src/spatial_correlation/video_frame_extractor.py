import cv2
import os
import shutil

if __name__ == "__main__":
    input_dir = "../../dataset/smu_videos/pi_3"
    video_name = "rpi_3_2019-03-28_18_04_16.mp4"

    # output directory for frames
    frames_dir = "{}/{}".format(input_dir, "frames")
    if os.path.exists(frames_dir):
        print("{} already exists, deleting..".format(frames_dir))
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)

    # extract frames
    input_video = cv2.VideoCapture("{}/{}".format(input_dir, video_name))
    assert input_video is not None

    frame_num = 1
    status, frame = input_video.read()
    while status:
        cv2.imwrite("{}/frame_{}.jpg".format(frames_dir, frame_num), frame)
        frame_num += 1
        status, frame = input_video.read()
        # break

    input_video.release()
