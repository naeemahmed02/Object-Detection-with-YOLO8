######### Import Required Libraries
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

######### Source
cap = cv.VideoCapture('resources/bikes.mp4')

######## Define the model
model = YOLO('wieghts/yolov8n.pt')

######## Looping the Source Frame by Frame
while True:
    _, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  #### Getting the Coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  #### Converting the Coordiante values into int
            cls = int(box.cls[0])   #### Getting class index
            class_name = model.names[cls]   #### Using class index getting the class name
            conf = math.ceil((box.conf[0] * 100))/100   #### Getting Predicted Confidence
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, bbox=(x1, y1, w, h))
            cvzone.putTextRect(img, f"{class_name}{' '}{conf}", (max(0,x1), max(35, y1)), scale=0.7, thickness=1, offset=3)

    cv.imshow("Car Counter", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

