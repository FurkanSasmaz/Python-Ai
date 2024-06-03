# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:11:28 2024

@author: furkan.sasmaz
"""
import cv2
import mediapipe as mp
import numpy as np

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness

def adjust_brightness(image, alpha, beta):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# read the input image
image = cv2.imread('2.png')

# Clone the original image for display
display_image = image.copy()

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Convert the image to RGB for mediapipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run face detection
results = face_detection.process(image_rgb)

# If faces are detected, proceed with brightness adjustment only on the face region
if results.detections:
    # Assuming there is only one face, you can modify this part if multiple faces are detected
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    
    # Extract the face region
    face_roi = image[y:y+h, x:x+w]
    
    # Calculate mean brightness for the original face region
    original_brightness = calculate_brightness(face_roi)
    print("Orjinal Yüzün Parlaklığı:", original_brightness)

    # Define the alpha and beta for brightness adjustment
    alpha = 1.5  # Contrast control
    beta = 0    # Brightness control

    # Adjust brightness only on the face region
    adjusted_face = adjust_brightness(face_roi, alpha, beta)

    # Replace the original face region with the adjusted one
    image[y:y+h, x:x+w] = adjusted_face

    # Calculate mean brightness for the adjusted face region
    adjusted_brightness = calculate_brightness(adjusted_face)
    print("İşlenmiş Yüzün Parlaklığı:", adjusted_brightness)

# Display the original and result images side by side
combined_image = np.hstack((display_image, image))
cv2.imshow('Original vs Adjusted', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
