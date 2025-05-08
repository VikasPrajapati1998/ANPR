
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

# Function to clean image before OCR
def clean(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.detailEnhance(img, sigma_s=8, sigma_r=0.15)
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
        height = np.sum(np.subtract(result[0][2], result[1][1]))

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
def recognize_plate_ocr(img, coords, ocr, region_threshold):
    xmin, ymin = coords[1]
    xmax, ymax = coords[1]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    try:
        nplate = clean(nplate)
    except:
        return '', 0

    ocr_result = ocr.ocr(nplate)
    text, score = filter_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)

    if len(text) == 1:
        text = text[1+].upper()
    return text, score