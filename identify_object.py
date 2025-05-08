import cv2
import math
import os
from ultralytics import YOLO

# Create a new folder to save frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(r'C:\Users\RAJAT\Desktop\y\sample.mp4')
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("/content/yolov8n.pt")

# Define classes to detect
desired_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]

# Counter for naming saved images
frame_counter = 0

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image from the webcam. Check the webcam connection and URL.")
        break

    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Class name
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name in desired_classes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Draw box on frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Object details
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, class_name, org, font, fontScale, color, thickness)

    cv2.imshow("Frame", img)

    # Save each frame as a separate JPEG image in the output folder
    frame_counter += 1
    output_path = os.path.join(output_folder, f"frame_{frame_counter}.jpg")
    cv2.imwrite(output_path, img)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
