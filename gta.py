from datetime import datetime
import logging
from typing import List, Optional, Union
import time
import requests
import socket 
import argparse
import os
import re
import threading 

import cv2
import numpy as np
import pandas as pd
import torch
import norfair
from norfair import Detection, Tracker
from paddleocr import PaddleOCR
import albumentations as A
from shapely.geometry import Point, Polygon
import sqlite3
from flask import Flask, render_template, Response, jsonify

# Global variables and constants
LOG_FILE = "plate_log.txt"
DISTANCE_THRESHOLD_BBOX: float = 0.7
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1

# Define your polygon
Poligono = [[200, 485], [200, 655], [1250, 655], [1250, 485]]
polygon1 = Polygon(Poligono)

# Database setup
conn = sqlite3.connect('plate_numbers.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS plates
             (timestamp TEXT,
              plate_number TEXT UNIQUE,
              camera TEXT)''')
conn.commit()

# Function to clean image before OCR
def clean(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img

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
        img = clean(img)
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

# Save detected plate number along with timestamp to the database
def save_to_db(timestamp: str, plate_number: str, camera: str):
    try:
        c.execute("INSERT INTO plates (timestamp, plate_number, camera) VALUES (?, ?, ?)", (timestamp, plate_number, camera))
        conn.commit()
        if camera == "entry":  # Trigger relay for entry gate
            control_relay(b'RELAY_ON')  # Turn on the relay
            time.sleep(2)  # Delay for 2 seconds (adjust as needed)
            control_relay(b'RELAY_OFF')  # Turn off the relay
    except sqlite3.IntegrityError:
        print(f"Plate number {plate_number} already exists in the database.")

# Control relay using socket connection
def control_relay(command):
    try:
        # Define the socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('192.168.1.37', 5005))
            # Send a command to control the relay
            s.sendall(command)
            # Optionally, receive a response from the relay
            data = s.recv(1024)
            print(f"Received response from relay: {data}")
    except Exception as e:
        print(f"Error controlling relay: {e}")

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
                            save_to_db(timestamp, license_plate, camera_name)
                            draw_label(frame, license_plate, points[0][0], points[0][1])
                            df = process_ocr_result(frame, points, license_plate, df, output_folder)

    return frame

# Process OCR results, update dataframe, and save images
def process_ocr_result(frame, points, plate_number, df, output_folder):
    plate_dict = {"Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], "Plate Number": [plate_number]}
    df = pd.concat([df, pd.DataFrame(plate_dict)], ignore_index=True)
    output_path = os.path.join(output_folder, f"{plate_number}.jpg")
    cv2.imwrite(output_path, frame[points[0][1]:points[1][1], points[0][0]:points[1][0]])
    return df

# Main function to process video and detect plates
def main(args):
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    model = YOLO(weights=args.weight, conf_threshold=args.conf_threshold)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    tracker = Tracker(distance_function=norfair.distances.iou, distance_threshold=DISTANCE_THRESHOLD_BBOX)

    cap_entry = cv2.VideoCapture(args.input_entry)
    cap_exit = cv2.VideoCapture(args.input_exit)
    if not cap_entry.isOpened():
        raise Exception(f"Error opening video stream or file: {args.input_entry}")
    if not cap_exit.isOpened():
        raise Exception(f"Error opening video stream or file: {args.input_exit}")

    df_entry = pd.DataFrame(columns=["Timestamp", "Plate Number"])
    df_exit = pd.DataFrame(columns=["Timestamp", "Plate Number"])
    os.makedirs(args.output_folder, exist_ok=True)

    def process_video_stream(cap, df, camera_name):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, model, ocr, tracker, df, args.output_folder, args.region_threshold, camera_name)

        return df

    df_entry = process_video_stream(cap_entry, df_entry, "entry")
    df_exit = process_video_stream(cap_exit, df_exit, "exit")

    df_entry.to_excel(args.excel_entry, index=False)
    df_exit.to_excel(args.excel_exit, index=False)
    cap_entry.release()
    cap_exit.release()
    cv2.destroyAllWindows()

# Argument parsing and script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Number Plate Recognition (ANPR) system with entry and exit cameras.")
    parser.add_argument("--weight", type=str, required=True, help="Path to YOLOv5 weights file.")
    parser.add_argument("--input_entry", type=str, required=True, help="Path to input video file for entry camera.")
    parser.add_argument("--input_exit", type=str, required=True, help="Path to input video file for exit camera.")
    parser.add_argument("--excel_entry", type=str, required=True, help="Path to output Excel file for entry camera.")
    parser.add_argument("--excel_exit", type=str, required=True, help="Path to output Excel file for exit camera.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save detected plate images.")
    parser.add_argument("--conf_threshold", type=float, default=0.55, help="Confidence threshold for YOLOv5 model.")
    parser.add_argument("--region_threshold", type=float, default=0.2, help="Region threshold for OCR filtering.")

    args = parser.parse_args()
    main(args)
    
    # Flask web server for live monitoring
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_stream(cap, model, ocr, tracker, df, output_folder, region_threshold, camera_name):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, ocr, tracker, df, output_folder, region_threshold, camera_name)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed_entry')
def video_feed_entry():
    return Response(generate_video_stream(cv2.VideoCapture(args.input_entry), model, ocr, tracker, df_entry, args.output_folder, args.region_threshold, "entry"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_exit')
def video_feed_exit():
    return Response(generate_video_stream(cv2.VideoCapture(args.input_exit), model, ocr, tracker, df_exit, args.output_folder, args.region_threshold, "exit"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_plate_log')
def get_plate_log():
    c.execute("SELECT timestamp, plate_number, camera FROM plates ORDER BY timestamp DESC LIMIT 10")
    log_entries = [{"timestamp": row[0], "plate_number": row[1], "camera": row[2]} for row in c.fetchall()]
    return jsonify(log_entries)

if __name__ == '__main__':
    threading.Thread(target=main, args=(args,)).start()
    app.run(host='0.0.0.0', port=5000)