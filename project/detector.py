import cv2
import datetime
import os
import imageio
from pygifsicle import optimize
import math
import numpy as np
import sys

class Detector(object):
    def __init__(self, video):
        self.video = video
        self.average = None

    def detect(self):
        ret,frame=self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        result = {}

        # if the average frame is None, initialize it
        if self.average is None:
            print("[INFO] starting background model...")
            self.average = gray.copy().astype("float")
            return {}

        cv2.accumulateWeighted(gray, self.average, 0.5)
        average_abs = cv2.convertScaleAbs(self.average)
        deltaframe = cv2.absdiff(gray, average_abs)

        threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold,None)

        countour,heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_motion = False

        countour_count = 0
        countour_total_area = 0
        for i in countour:
            if cv2.contourArea(i) < 50:
                continue

            has_motion  = True
            (x, y, w, h) = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            countour_count += 1
            countour_total_area += w*h

        text = "{} | {}".format(countour_count, countour_total_area)
        write_on_frame(frame, text, (50, 50))

        result['countour_count'] = countour_count
        result['countour_total_area'] = countour_total_area
        result['has_motion'] = has_motion
        result['threshold'] = threshold
        result['deltaframe'] = deltaframe
        result['average_abs'] = average_abs
        result['frame'] = frame
        return result

def write_on_frame(frame, text, pos):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255,0,0), 2, cv2.LINE_AA)

def get_upload_path(suffix, file_extention):
    dt = datetime.datetime.now()
    folder = dt.strftime('G:\\My Drive\\archive\%Y%m%d')
    file = dt.strftime('%H%M%S_{}.{}'.format(suffix,file_extention))
    os.makedirs(os.path.join(folder), exist_ok=True)
    filename = os.path.join( folder, file)
    return filename

class Heartbeat(object):
    def __init__(self, heartbeat_delta_seconds):
        self.heartbeat_delta = datetime.timedelta(seconds=heartbeat_delta_seconds)
        self.last_heartbeat = datetime.datetime(1970,1,1)

    def check_upload(self, frame):

        if frame is None:
            return

        now = datetime.datetime.now()
        if self.last_heartbeat + self.heartbeat_delta < now:
            self.last_heartbeat = now
            path = get_upload_path('heartbeat', 'png')
            print(f"Writing heartbeat to {path}")
            cv2.imwrite(path, frame)

class MovementSignal(object):

    def __init__(self, rolling_count):
        self.rolling_count = rolling_count
        self.signal_up = False
        self.value_window = []

    def update_signal(self, value):

        self.value_window.append(value)
        self.value_window = self.value_window[-self.rolling_count:]
        rolling_average = self.get_average()
        self.signal_up = rolling_average > 0

    def get_average(self):
        return sum(self.value_window) / len(self.value_window)

    def get_signal(self):
        return self.signal_up

class FrameBuffer(object):
    def __init__(self, frame_save_delta_seconds):
        self.frame_save_delta = datetime.timedelta(seconds = frame_save_delta_seconds)
        self.last_save = datetime.datetime(1970, 1, 1)
        self.buffer = []

    def save(self, frame):
        now = datetime.datetime.now()
        if self.last_save + self.frame_save_delta < now and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.buffer.append(frame)

    def write(self, optimize=False, target_frames=150):
        if len(self.buffer) == 0:
            return

        buffer = self.buffer
        buff_len = len(buffer)
        self.buffer = []

        frame_factor = int(math.ceil(buff_len / float(target_frames)))
        if frame_factor > 1:
            buffer = buffer[::frame_factor]
            buff_len = len(buffer)

        for idx, frame in enumerate(buffer):
            write_on_frame(frame, "{}/{}".format(idx+1, buff_len), (50, 100))

        output_path = get_upload_path('motion', 'gif')
        write_path = 'temp.gif' if optimize else output_path

        print(f"Saving {buff_len} Frames ...")
        imageio.mimsave(write_path, buffer, duration = 0.1)
        if optimize:
            optimize(write_path,  output_path)
        print(f"Writing motion to {output_path}")
        self.last_save = datetime.datetime(1970, 1, 1)


class ImageStitcher(object):
    def __init__(self, widths, heights, scale):
        self.widths = widths
        self.heights = heights
        self.scale = scale
        self.canvas = None

    def combine_image(self, image, offset_h, offset_w):
        width = int(image.shape[1] * self.scale / 100)
        height = int(image.shape[0] * self.scale / 100)
        resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

        if self.canvas is None:
            self.canvas = np.zeros((height*self.heights, width*self.widths, 3), np.uint8)

        w_start = width * offset_w
        w_end = width * (offset_w+1)

        h_start = height * offset_h
        h_end = height * (offset_h+1)

        self.canvas[h_start:h_end, w_start:w_end, :3] = resized

cap=cv2.VideoCapture(0)
detector = Detector(cap)
heartbeat = Heartbeat(60* 60)
movement_signal = MovementSignal(45)
movement_signal2 = MovementSignal(45)
frame_buffer = FrameBuffer(2)

csv = []
while(True):

    result = detector.detect()

    if result.get('frame') is not None:
        image_stitcher = ImageStitcher(widths=2, heights=2, scale=50)
        image_stitcher.combine_image(result.get('frame'), offset_w=0, offset_h=0)
        image_stitcher.combine_image(cv2.cvtColor(result.get('average_abs'), cv2.COLOR_GRAY2RGB), offset_w=1, offset_h=0)
        image_stitcher.combine_image(cv2.cvtColor(result.get('deltaframe'), cv2.COLOR_GRAY2RGB), offset_w=0, offset_h=1)
        image_stitcher.combine_image(cv2.cvtColor(result.get('threshold'), cv2.COLOR_GRAY2RGB), offset_w=1, offset_h=1)
        cv2.imshow('all', image_stitcher.canvas)

    heartbeat.check_upload(result.get('frame'))
    movement_signal.update_signal(result.get('countour_count', 0))
    movement_signal2.update_signal(result.get('countour_total_area', 0))

    csv.append( (str(len(csv)),
                 str(result.get('countour_count', 0)),
                 str(movement_signal.get_average()),
                 str(result.get('countour_total_area', 0)),
                 str(movement_signal2.get_average()),
                 ))


    if movement_signal.get_signal():
        pass#frame_buffer.save(result.get('frame'))
    else:
        pass#frame_buffer.write()

    if cv2.waitKey(20) == ord('q'):
      break

print("idx,", 'contour_count,', 'contour_count_rolling,', 'countour_total_area', 'countour_total_area_rolling,')
print("\n".join( ",".join(l) for l in csv ))
cap.release()
cv2.destroyAllWindows()