import cv2



def doit(cap, average):
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    result = {}

	# if the average frame is None, initialize it
    if average is None:
        print("[INFO] starting background model...")
        average = gray.copy().astype("float")
        return {'average': average}
    result['average'] = average

    cv2.accumulateWeighted(gray, average, 0.5)
    deltaframe = cv2.absdiff(gray, cv2.convertScaleAbs(average))


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

average = None
cap=cv2.VideoCapture(0)
while(True):

    result = doit(cap, average)

    average = result.get('average')

    for key in ['threshold', 'deltaframe', 'frame']:
        img = result.get(key)
        if img is not None:
            cv2.imshow(key, img)


    if cv2.waitKey(20) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()