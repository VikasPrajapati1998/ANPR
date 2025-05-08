import threading
import time
from datetime import datetime
import logging
from typing import List, Optional, Union
import requests
import argparse
import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
import norfair
from norfair import Detection, Tracker
from paddleocr import PaddleOCR
import albumentations as A
from shapely.geometry import Point, Polygon
from PIL import Image, ImageEnhance, ImageFilter
from threading import Lock

# Global variables and constants
LOG_FILE = "plate_log.txt"
DISTANCE_THRESHOLD_BBOX: float = 0.7
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1
LOCK = Lock()  # Lock for thread-safe operations on shared resources

# Define your polygon
Poligono = [[200, 485], [200, 655], [1250, 655], [1250, 485]]
polygon1 = Polygon(Poligono)

# Function to clean image before OCR
def clean_with_pil(img):
    pil_img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(2)
    pil_img = pil_img.filter(ImageFilter.MedianFilter())
    open_cv_image = np.array(pil_img)
    return open_cv_image

# Function to filter text based on region threshold
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate, scores = [], []
    for result in ocr_result[0]:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1][0])
            scores.append(result[1][1])

    plate = ''.join(plate)
    plate = re.sub(r'\W+', '', plate)
    if not scores:
        plate = ''
        scores.append(0)
    return plate.upper(), max(scores)

# Function to recognize plate numbers using OCR
def recognize_plate_ocr(img, ocr, region_threshold):
    try:
        img = clean_with_pil(img)
    except Exception as e:
        logging.error(f"Error cleaning image: {e}")
        return '', 0

    ocr_result = ocr.ocr(img)
    text, score = filter_text(region=img, ocr_result=ocr_result, region_threshold=region_threshold)

    if len(text) == 1:
        text = text[0].upper()
    return text, score

# Class for YOLOv5 model
class YOLO:
    def __init__(self, weights, device: Optional[str] = None, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load('./yolov5', 'custom', source='local', path=weights, force_reload=True)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

    def __call__(self, img: Union[str, np.ndarray], image_size: int = 640, classes: Optional[List[int]] = None) -> torch.tensor:
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections

# Convert YOLO detections to norfair detections
def yolo_detections_to_norfair_detections(yolo_detections: torch.tensor):
    norfair_detections = []
    detections_as_xyxy = yolo_detections.xyxy[0]
    for detection_as_xyxy in detections_as_xyxy:
        bbox = np.array(
            [
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ]
        )
        scores = np.array(
            [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
        )
        norfair_detections.append(
            Detection(
                points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
            )
        )

    return norfair_detections

# Data augmentation transformations
augmentations = A.Compose([
    A.Rotate(limit=10, p=0.5),  
    A.RandomBrightnessContrast(p=0.5),  
    A.HorizontalFlip(p=0.5),  
])

# Apply data augmentation to an image and its bounding box coordinates
def apply_augmentation(image, bounding_box):
    augmented = augmentations(image=image, bboxes=[bounding_box], bbox_params=A.BboxParams(format='pascal_voc'))
    return augmented['image'], augmented['bboxes'][0]

# Draw label onto image
def draw_label(im, label, x, y):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y), (0, 0, 0), cv2.FILLED)
    cv2.putText(im, label, (x, y - baseline), FONT_FACE, FONT_SCALE, (0, 255, 0), THICKNESS, cv2.LINE_AA)

# Log detected plate number along with timestamp to a text file
def log_plate_detection(timestamp: str, plate_number: str, camera: str):
    with open(LOG_FILE, "a") as file:
        file.write(f"{timestamp}: {plate_number} ({camera})\n")

# Define the HTTP POST function function for http post the data 
def post_vehicle_data(plate_image_path, plate_number, vehicle_type):
    url = 'http://127.0.0.1:8000//vehicle_entry'  # Define your URL here
    max_attempts = 5
    attempt = 0
    POST_RETRY_DELAY = 5  # seconds

    while attempt < max_attempts:
        try:
            with open(plate_image_path, 'rb') as plate_image_file:
                data = {
                    'vehicle_number': plate_number,
                    'vehicle_type': vehicle_type
                }
                files = {
                    'vehicle_image': plate_image_file
                }
                response = requests.post(url, data=data, files=files)

                if response.status_code == 201:
                    print('Data posted successfully')
                    return response
                else:
                    print(f'Failed to post data: {response.status_code} - {response.text}')
                    attempt += 1
                    time.sleep(POST_RETRY_DELAY)
        except Exception as e:
            print(f'Error posting data: {e}')
            attempt += 1
            time.sleep(POST_RETRY_DELAY)
    return None

# Function to check if plate is within the region of interest
def is_plate_in_roi(points):
    center_x = (points[0][0] + points[1][0]) / 2
    center_y = (points[0][1] + points[1][1]) / 2
    return polygon1.contains(Point(center_x, center_y))

# Process each frame
def process_frame(frame, model, ocr, tracker, df, output_folder, region_threshold, camera_name):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw ROI polygon on frame
    cv2.polylines(frame, [np.array(Poligono)], isClosed=True, color=(255, 0, 0), thickness=2)

    yolo_detections = model(frame)
    if yolo_detections is not None:
        detections = yolo_detections_to_norfair_detections(yolo_detections)
        if detections:
            tracked_objects = tracker.update(detections=detections)
            if tracked_objects is not None:
                for obj in tracked_objects:
                    points = obj.estimate.astype(int)
                    points = tuple(points)

                    if is_plate_in_roi(points):
                        cv2.rectangle(frame, points[0], points[1], (0, 255, 0), 2)
                        vehicle_img = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]
                        license_plate, score = recognize_plate_ocr(vehicle_img, ocr, region_threshold)
                        
                        if license_plate:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_plate_detection(timestamp, license_plate, camera_name)
                            # Save the vehicle image temporarily
                            vehicle_img_path = f'{output_folder}/{license_plate}_{timestamp}.jpg'
                            cv2.imwrite(vehicle_img_path, vehicle_img)
                            post_vehicle_data(vehicle_img_path, license_plate, 'Vehicle')

                            with LOCK:
                                if license_plate not in df.index:
                                    df.loc[license_plate] = 1
                                else:
                                    df.loc[license_plate] += 1
                                df.to_csv('plate_counts.csv')
                            draw_label(frame, f'{license_plate} ({df.loc[license_plate][0]})', points[0][0], points[0][1])

    return frame

# Capture video from a given RTSP URL
def capture_video(rtsp_url, camera_name, model, ocr, tracker, df, output_folder, region_threshold):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"Unable to open video capture for {camera_name}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Failed to capture frame from {camera_name}")
            time.sleep(5)  # Wait before retrying
            cap = cv2.VideoCapture(rtsp_url)  # Reinitialize capture
            continue

        frame = process_frame(frame, model, ocr, tracker, df, output_folder, region_threshold, camera_name)
        cv2.imshow(camera_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main(rtsp_urls, camera_names, weights_path, output_folder='captured_plates', region_threshold=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load YOLO model
    model = YOLO(weights=weights_path)

    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Initialize Norfair tracker
    tracker = Tracker(distance_function=norfair.distances.iou, distance_threshold=DISTANCE_THRESHOLD_BBOX)

    # DataFrame to store license plate counts
    if os.path.exists('plate_counts.csv'):
        df = pd.read_csv('plate_counts.csv', index_col=0)
    else:
        df = pd.DataFrame(columns=['count'])

    # Create threads for each camera
    threads = []
    for rtsp_url, camera_name in zip(rtsp_urls, camera_names):
        thread = threading.Thread(target=capture_video, args=(rtsp_url, camera_name, model, ocr, tracker, df, output_folder, region_threshold))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANPR with multiple RTSP streams')
    parser.add_argument('--rtsp-urls', type=str, nargs='+', required=True, help='List of RTSP URLs')
    parser.add_argument('--camera-names', type=str, nargs='+', required=True, help='List of camera names corresponding to RTSP URLs')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv5 weights')
    args = parser.parse_args()
    main(args.rtsp_urls, args.camera_names, args.weights)
