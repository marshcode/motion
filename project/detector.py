import cv2


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
        deltaframe = cv2.absdiff(gray, cv2.convertScaleAbs(self.average))


        threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold,None)

        countour,heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in countour:
            if cv2.contourArea(i) < 50:
                continue

            (x, y, w, h) = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        result['threshold'] = threshold
        result['deltaframe'] = deltaframe
        result['frame'] = frame
        return result


cap=cv2.VideoCapture(0)
detector = Detector(cap)
while(True):

    result = detector.detect()

    average = result.get('average')

    for key in ['threshold', 'deltaframe', 'frame']:
        img = result.get(key)
        if img is not None:
            cv2.imshow(key, img)


    if cv2.waitKey(20) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()