"""
module to capture hi-resolution data collection (using picamera lib) for training of coordinate mapper and overlap estimator
"""

from picamera import PiCamera
import cv2
import time
from multiprocessing import Process
from datetime import datetime as dt
import multiprocessing

FLAG = multiprocessing.Value('b', 0)


def capture_images():
    with PiCamera() as camera:
        camera.resolution = (1056, 1056)
        time.sleep(2)
        print("Starting capture \n")
        while (FLAG.value == 0):
            dt_now = str(dt.now()).replace(" ", "-").replace(":", "-")
            img_name = "images/" + str(dt_now) + ".jpg"
            camera.capture(img_name, use_video_port=True)
        print("Closing Camera\n")


if __name__ == "__main__":
    # initialize and run camera
    process = Process(target=capture_images)
    process.start()

    print("Press q to quit\n")
    while (1):
        key = input()
        print("Key is -- {}\n".format(key))
        if key == "q":
            print("Quitting\n")
            FLAG.value = 1
            break
