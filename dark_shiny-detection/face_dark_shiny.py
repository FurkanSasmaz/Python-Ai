# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 09:48:11 2023
@author: furkan.sasmaz
"""
import cv2
import mediapipe as mp

# Kamera akışını başlat
cap = cv2.VideoCapture(0)

# Mediapipe yüz tespiti için gerekli modülü yükle
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Yüz tespiti modülünü başlat
face_detection = mp_face_detection.FaceDetection()

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()

    # Eğer kare okunamadıysa döngüyü sonlandır
    if not ret:
        break

    # Kareyi BGR renk formatından RGB'ye dönüştür
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # RGB görüntüyü Mediapipe'ın girdi formatına dönüştür
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            # Yüz bölgesini belirle
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            face_roi = frame[y:y+height, x:x+width]

            # Yüz bölgesindeki ışık miktarını hesapla
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            light_level = cv2.mean(gray)[0]
                # Görüntü çok karanlık mı?
            if light_level < 50:
                cv2.putText(frame, "Too Dark", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("karanlık")
                # Görüntü çok parlak mı?
            if light_level > 120:
                cv2.putText(frame, "Too Shiny", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Parlak")
            # Yüz bölgesini çerçeve içine al
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

            # Işık miktarını ekranda göster
            cv2.putText(frame, f'Light Level: {light_level:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # İşlenmiş kareyi ekranda göster
    cv2.imshow('Frame', frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera akışını durdur ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()