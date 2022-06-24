import cv2
import datetime
import os


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

cap=cv2.VideoCapture(0)
detector = Detector(cap)
heartbeat = Heartbeat(5)

while(True):

    result = detector.detect()

    for key in ['threshold', 'deltaframe', 'frame', 'average_abs']:
        img = result.get(key)
        if img is not None:
            cv2.imshow(key, img)

    heartbeat.check_upload(result.get('frame'))

    if cv2.waitKey(20) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()