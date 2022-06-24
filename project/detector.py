import cv2
import datetime
import os
import imageio
from pygifsicle import optimize
import math

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
        cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255,0,0), 2, cv2.LINE_AA)

        result['countour_count'] = countour_count
        result['countour_total_area'] = countour_total_area
        result['has_motion'] = has_motion
        result['threshold'] = threshold
        result['deltaframe'] = deltaframe
        result['average_abs'] = average_abs
        result['frame'] = frame
        return result

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
        rolling_average = sum(self.value_window) / len(self.value_window)
        self.signal_up = rolling_average > 0

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
        frame_factor = int(math.ceil(len(self.buffer) / float(target_frames)))
        if frame_factor > 1:
            buffer = buffer[::frame_factor]

        output_path = get_upload_path('motion', 'gif')
        write_path = 'temp.gif' if optimize else output_path

        print(f"Saving {len(buffer)} Frames ...")
        imageio.mimsave(write_path, buffer, duration = 0.1)
        if optimize:
            optimize(write_path,  output_path)
        print(f"Writing motion to {output_path}")
        self.buffer = []
        self.last_save = datetime.datetime(1970, 1, 1)

cap=cv2.VideoCapture(0)
detector = Detector(cap)
heartbeat = Heartbeat(60* 60)
movement_signal = MovementSignal(100)
frame_buffer = FrameBuffer(2)

while(True):

    result = detector.detect()

    for key in ['threshold', 'deltaframe', 'frame', 'average_abs']:
        img = result.get(key)
        if img is not None:
            cv2.imshow(key, img)

    heartbeat.check_upload(result.get('frame'))
    movement_signal.update_signal(result.get('countour_count', 0))

    if movement_signal.get_signal():
        frame_buffer.save(result.get('frame'))
    else:
        frame_buffer.write()

    if cv2.waitKey(20) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()