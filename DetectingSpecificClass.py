############ Import Required Libraries
from ultralytics import YOLO
import cvzone
import cv2 as cv
import math

############ Capture the video, webcam or any media source
cap = cv.VideoCapture('resources/cars.mp4')
cap.set(3, 640)
cap.set(4, 640)

############ Model for Detections
model = YOLO('wieghts/yolov8m.pt')

############ Loop through Frame by Frame
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]   ##### Getting the coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)   ##### Converting the coordinates into int
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))
            cls = int(box.cls[0])   ##### Getting Class index
            class_names = model.names[cls] ##### Getting Class Names
            conf = math.ceil((box.conf[0] * 100))/100  ##### Getting Confidence Level
            if class_names == "car" and "truck" and "motorbike" and "bus" and conf > .35:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                cvzone.putTextRect(img, f"{class_names}{' '}{conf}", (max(0, x1), max(0, y1)), scale=0.6, thickness=1, offset=3)
        cv.imshow("image", img)




    cv.waitKey(1) & 0xFF == ord('q')

cv.destroyAllWindows()

