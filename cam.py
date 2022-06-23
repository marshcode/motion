# python -m pip install numpy
# python -m pip install opencv-python

# importing OpenCV library
import cv2
import shutil
import time
import datetime
import os

def take_and_save(filename):
    # initialize the camera
    # If you have multiple camera connected with
    # current device, assign a value in cam_port
    # variable according to that
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    result = None
    try:
        # reading the input using the camera
        result, image = cam.read()
    except Exception as e:
        pass
    finally:
        cam.release()

    if result:
        # saving image in local storage
        print("Writing to {}".format(filename))
        cv2.imwrite(filename, image)
    else:
        print("Error: Writing failure to {}".format(filename))
        shutil.copy('error.png', filename)


while True:
    dt = datetime.datetime.now()
    folder = dt.strftime('G:\\My Drive\\archive\%Y%m%d')
    file = dt.strftime('%H%M%S.png')
    os.makedirs(os.path.join('archive', folder), exist_ok=True)
    take_and_save(os.path.join('archive', folder, file))

    time.sleep(5 * 60)
