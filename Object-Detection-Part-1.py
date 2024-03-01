from ultralytics import YOLO
import cv2 as cv
import math
import cvzone

############## Camera Setting ##############
# cap = cv.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv.VideoCapture('re4')

############## YOLO Model
model = YOLO('wieghts\yolov8l.pt')

############## Looping
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2,y2 = box.xyxy[0]
            x1, y1, x2,y2 = int(x1), int(y1), int(x2), int(x2)
            w, h = x2 - x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            ######### Getting Class Names
            cls = int(box.cls[0])
            class_name = model.names[cls]
            ### Getting Confidence value
            conf = math.ceil((box.conf[0] * 100))/100
            cvzone.putTextRect(img, f"{class_name}{' '}{conf}", (max(0, x1), max(0, y1)), scale=0.6, thickness=1)

    cv.imshow("Object Detection", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()

    