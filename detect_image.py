from ultralytics import YOLO 
import cv2
import os 

print(os.path.exists("_SR73139.jpg"))

model = YOLO("yolov8n.pt")
results = model("/Users/perzyy/Documents/Object-detection /Object-detection/venv/_SR73139.jpg")

annotated_image = results[0].plot()  # Draw bounding boxes on the image
cv2.imshow("Detection", annotated_image)
cv2.waitKey(0) 