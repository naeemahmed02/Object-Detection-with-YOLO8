##### Object Tracking

##### Import Libraries
import cv2 as cv
import cvzone

##### Initializing Variable
FRAME_WIDTH = 640
FRAME_HEIGHT = 400

##### Define Source
cap = cv.VideoCapture(1)  # Capture video from webcam (assuming webcam index 1)
cap.set(3, FRAME_WIDTH)   # Set frame width
cap.set(4, FRAME_HEIGHT)  # Set frame height

##### Define Tracker
tracker = cv.legacy.TrackerMOSSE_create()  # Create MOSSE tracker
success, img = cap.read()  # Read the first frame from the webcam
bbox = cv.selectROI("tracker", img, False)  # Select ROI (Region of Interest) for tracking
tracker.init(img, bbox)  # Initialize the tracker with the selected ROI

def drawbbox(img, bbox):
    # Extract coordinates and dimensions of the bounding box
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # Draw rectangle around the tracked object
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
    # Add text indicating "Tracking" near the object
    cvzone.putTextRect(img, "Tracking", (75, 120), scale=1, thickness=1)
    print(x)  # Print x-coordinate of the object

##### Looping the Frame by Frame
while True:
    timer = cv.getTickCount()  # Get the number of clock ticks
    _, img = cap.read()  # Read frames from the webcam
    success, bbox = tracker.update(img)  # Update the tracker with the current frame

    if success:
        drawbbox(img, bbox)  # If tracking is successful, draw the bounding box around the object
    else:
        cvzone.putTextRect(img, "Lost", (75, 120), scale=0.6, thickness=1)  # If tracking is lost, display "Lost" message

    fps = cv.getTickFrequency() / (cv.getTickCount()-timer)  # Calculate frames per second (FPS)
    cvzone.putTextRect(img, f"FPS {str(int(fps))}", (75, 50), scale=0.6, thickness=1, font=cv.FONT_HERSHEY_SCRIPT_SIMPLEX)
    cv.imshow("Demo", img)  # Display the frame with tracking information

    if cv.waitKey(1) & 0xff == ord('q'):  # Break the loop if 'q' is pressed
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
