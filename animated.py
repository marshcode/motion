import cv2
import imageio
# 1. Get Images
cap = cv2.VideoCapture(0)

frames = []
image_count = 0
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(0)
    if key == ord("a"):
        image_count += 1
        frames.append(frame)
        print("Adding new image:", image_count)
    elif key == ord("q"):
        break
print("Images added: ", len(frames))

imageio.mimsave('my_very_own_gif.gif', frames, duration = 0.25)
