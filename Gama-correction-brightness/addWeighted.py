# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:26:06 2024

@author: furkan.sasmaz
"""
import cv2
import mediapipe as mp
import numpy as np

# Load Mediapipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness

# Read image
img = cv2.imread('2.png')

# Face detection
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detection.process(img_rgb)

# if face_detection
if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih)

        # compy original image
        img_copy = img.copy()

        # for face area
        face_region_original = img[y:y+h, x:x+w]

        # Kontrast ve parlaklık ayarını uygula
        contrast = 5.0  # Kontrast kontrolü (0'dan 127'ye kadar)
        brightness = 2.0  # Parlaklık kontrolü (0-100)
        face_region_adjusted = cv2.addWeighted(face_region_original, contrast, face_region_original, 0, brightness)

        # Orijinal resmi güncellenmiş yüz bölgesi ile değiştir
        img_copy[y:y+h, x:x+w] = face_region_adjusted
        
        original_brightness = calculate_brightness(face_region_original)
        adjusted_brightness = calculate_brightness(face_region_adjusted)
        print("Orjinal Yüzün Parlaklığı:", original_brightness)
        print("İşlenmiş Yüzün Parlaklığı:", adjusted_brightness)


# Orijinal resmi ve işlenmiş resmi göster
cv2.imshow('Original', img)
cv2.imshow('Adjusted', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


