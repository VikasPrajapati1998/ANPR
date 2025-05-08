"""import threading
import argparse
import cv2
import pandas as pd
import sqlite3
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory
from anpr import main, process_frame, YOLO, PaddleOCR, Tracker, DISTANCE_THRESHOLD_BBOX
import norfair
import os

# Initialize the Flask application
app = Flask(__name__)

# Database setup (move this part from anpr.py if necessary)
conn = sqlite3.connect('plate_numbers.db', check_same_thread=False)
c = conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_stream():
    cap = cv2.VideoCapture(args.input)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, ocr, tracker, df, args.output_folder, args.region_threshold, video_writer)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_plate_log')
def get_plate_log():
    c.execute("SELECT timestamp, plate_number FROM plates ORDER BY timestamp DESC LIMIT 10")
    log_entries = [{"timestamp": row[0], "plate_number": row[1]} for row in c.fetchall()]
    return jsonify(log_entries)

@app.route('/plates/<filename>')
def send_plate(filename):
    return send_from_directory(args.output_folder, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatic Number Plate Recognition (ANPR) system.")
    parser.add_argument("--weight", type=str, required=True, help="Path to YOLOv5 weights file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--excel", type=str, required=True, help="Path to output Excel file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save detected plate images.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video.")
    parser.add_argument("--conf_threshold", type=float, default=0.55, help="Confidence threshold for YOLOv5 model.")
    parser.add_argument("--region_threshold", type=float, default=0.2, help="Region threshold for OCR filtering.")

    args = parser.parse_args()
    
    model = YOLO(weights=args.weight, conf_threshold=args.conf_threshold)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    tracker = Tracker(distance_function=norfair.distances.iou, distance_threshold=DISTANCE_THRESHOLD_BBOX)
    
    df = pd.DataFrame(columns=["Timestamp", "Plate Number"])
    video_writer = None  # Ensure video_writer is defined here for the scope

    threading.Thread(target=main, args=(args,)).start()
    app.run(host='0.0.0.0', port=5000)
"""
import threading
import argparse
import cv2
import pandas as pd
import sqlite3
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory
from gta import main, process_frame, YOLO, PaddleOCR, Tracker, DISTANCE_THRESHOLD_BBOX
import norfair
import os

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output_folder'

# Database setup
conn = sqlite3.connect('plate_numbers.db', check_same_thread=False)
c = conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_stream(input_source, camera_name):
    print(f"Connecting to camera: {input_source}")
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_name}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera {camera_name}")
            break

        frame = process_frame(frame, model, ocr, tracker, df, args.output_folder, args.region_threshold, camera_name)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print(f"Camera stream {camera_name} ended")

@app.route('/video_feed_entry')
def video_feed_entry():
    return Response(generate_video_stream(args.input_entry, "entry"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_exit')
def video_feed_exit():
    return Response(generate_video_stream(args.input_exit, "exit"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_plate_log')
def get_plate_log():
    c.execute("SELECT timestamp, plate_number, camera FROM plates ORDER BY timestamp DESC LIMIT 10")
    log_entries = [{"timestamp": row[0], "plate_number": row[1], "camera": row[2]} for row in c.fetchall()]
    return jsonify(log_entries)

@app.route('/plates/<filename>')
def send_plate(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatic Number Plate Recognition (ANPR) system.")
    parser.add_argument("--weight", type=str, required=True, help="Path to YOLOv5 weights file.")
    parser.add_argument("--input_entry", type=str, required=True, help="RTSP URL for entry camera.")
    parser.add_argument("--input_exit", type=str, required=True, help="RTSP URL for exit camera.")
    parser.add_argument("--excel_entry", type=str, required=True, help="Path to output Excel file for entry camera.")
    parser.add_argument("--excel_exit", type=str, required=True, help="Path to output Excel file for exit camera.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save detected plate images.")
    parser.add_argument("--conf_threshold", type=float, default=0.55, help="Confidence threshold for YOLOv5 model.")
    parser.add_argument("--region_threshold", type=float, default=0.2, help="Region threshold for OCR filtering.")

    args = parser.parse_args()
    
    model = YOLO(weights=args.weight, conf_threshold=args.conf_threshold)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    tracker = Tracker(distance_function=norfair.distances.iou, distance_threshold=DISTANCE_THRESHOLD_BBOX)
    
    df = pd.DataFrame(columns=["Timestamp", "Plate Number", "Camera"])
    threading.Thread(target=main, args=(args,)).start()
    app.run(host='0.0.0.0', port=5000)


"""import threading
import argparse
import cv2
import pandas as pd
import mysql.connector
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory
from anpr import main, process_frame, YOLO, PaddleOCR, Tracker, DISTANCE_THRESHOLD_BBOX
import norfair
import os
from mysql.connector import Error

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output_folder'

# Database setup
db_config = {
    'user': 'root',
    'password': ' ',  # Replace with your actual MySQL password
    'host': 'localhost',
    'database': 'lpr'
}

# Create a connection to the database
try:
    conn = mysql.connector.connect(**db_config)
    c = conn.cursor()
    print("Database connection successful")
except Error as e:
    print(f"Error connecting to MySQL: {e}")

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_stream(input_source, camera_name):
    print(f"Connecting to camera: {input_source}")
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_name}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera {camera_name}")
            break

        frame = process_frame(frame, model, ocr, tracker, df, args.output_folder, args.region_threshold, camera_name)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print(f"Camera stream {camera_name} ended")

@app.route('/video_feed_entry')
def video_feed_entry():
    return Response(generate_video_stream(args.input_entry, "entry"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_exit')
def video_feed_exit():
    return Response(generate_video_stream(args.input_exit, "exit"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_plate_log')
def get_plate_log():
    c.execute("SELECT timestamp, plate_number, camera FROM plates ORDER BY timestamp DESC LIMIT 10")
    log_entries = [{"timestamp": row[0], "plate_number": row[1], "camera": row[2]} for row in c.fetchall()]
    return jsonify(log_entries)

@app.route('/plates/<filename>')
def send_plate(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatic Number Plate Recognition (ANPR) system.")
    parser.add_argument("--weight", type=str, required=True, help="Path to YOLOv5 weights file.")
    parser.add_argument("--input_entry", type=str, required=True, help="RTSP URL for entry camera.")
    parser.add_argument("--input_exit", type=str, required=True, help="RTSP URL for exit camera.")
    parser.add_argument("--excel_entry", type=str, required=True, help="Path to output Excel file for entry camera.")
    parser.add_argument("--excel_exit", type=str, required=True, help="Path to output Excel file for exit camera.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save detected plate images.")
    parser.add_argument("--conf_threshold", type=float, default=0.55, help="Confidence threshold for YOLOv5 model.")
    parser.add_argument("--region_threshold", type=float, default=0.2, help="Region threshold for OCR filtering.")

    args = parser.parse_args()
    
    model = YOLO(weights=args.weight, conf_threshold=args.conf_threshold)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    tracker = Tracker(distance_function=norfair.distances.iou, distance_threshold=DISTANCE_THRESHOLD_BBOX)
    
    df = pd.DataFrame(columns=["Timestamp", "Plate Number", "Camera"])
    threading.Thread(target=main, args=(args,)).start()
    app.run(host='0.0.0.0', port=5000)
"""

