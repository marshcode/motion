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
        for i in countour:
            if cv2.contourArea(i) < 50:
                continue

            has_motion  = True
            (x, y, w, h) = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        result['has_motion'] = has_motion
        result['threshold'] = threshold
        result['deltaframe'] = deltaframe
        result['average_abs'] = average_abs
        result['frame'] = frame
        return result


class UploadChecker(object):
    def __init__(self, heartbeat_delta_seconds, movement_delta_seconds):
        self.heartbeat_delta = datetime.timedelta(seconds=heartbeat_delta_seconds)
        self.movement_delta = datetime.timedelta(seconds=movement_delta_seconds)

        self.last_heartbeat_upload = datetime.datetime(2000, 1, 1)
        self.last_movement_upload = datetime.datetime(2000, 1, 1)

    def check(self, result):

        if not result:
            return False

        now = datetime.datetime.now()
        if self.last_heartbeat_upload + self.heartbeat_delta  < now:
            self.last_heartbeat_upload = now
            return True

        if self.last_movement_upload + self.movement_delta < now and result.get('has_motion'):
            self.last_movement_upload = now
            return True

        return False


def write(result):
    dt = datetime.datetime.now()
    folder = dt.strftime('G:\\My Drive\\archive\%Y%m%d')
    suffix = 'motion' if result.get('has_motion') else 'heartbeat'
    file = dt.strftime('%H%M%S_{}.png'.format(suffix))
    os.makedirs(os.path.join(folder), exist_ok=True)
    filename = os.path.join( folder, file)

    result = cv2.imwrite(filename, result.get('frame'))
    print("Writing to {} = {}".format(filename, result))

cap=cv2.VideoCapture(0)
detector = Detector(cap)
upload_checker = UploadChecker(heartbeat_delta_seconds=60*60, movement_delta_seconds=5)
while(True):

    result = detector.detect()

    for key in ['threshold', 'deltaframe', 'frame', 'average_abs']:
        img = result.get(key)
        if img is not None:
            cv2.imshow(key, img)


    if upload_checker.check(result):
        write(result)

    if cv2.waitKey(20) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()