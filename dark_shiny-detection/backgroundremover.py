# import cv2
# import cvzone
# from cvzone.SelfiSegmentationModule import SelfiSegmentation


# # Kamera kaynağı veya video dosyası
# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# segmentor = SelfiSegmentation()

# imgBg = cv2.imread("white_1.jpg")

# while True:
#     # Kameradan görüntü yakala
#     ret, frame = cap.read()

#     # Görüntüyü gri tona dönüştür
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     imgOut = segmentor.removeBG(frame, imgBg, threshold=0.3)
    
#     imgStacked = cvzone.stackImages([imgOut, frame], 2, 1)
#     # Görüntünün parlaklığını hesapla
#     brightness = cv2.mean(gray)[0]
    
#     # Görüntü çok karanlık mı?
#     if brightness < 50:
#         cv2.putText(imgStacked, "Too Dark", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
#     # Görüntü çok parlak mı?
#     if brightness > 120:
#         cv2.putText(imgStacked, "Too Shiny", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     print(brightness)
#     # Görüntüyü ekranda göster
    
#     cv2.imshow("frame", imgStacked)

#     # Q tuşuna basarak çıkış yap
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Kaynakları serbest bırak
# cap.release()
# cv2.destroyAllWindows()



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
            print(light_level)
                # Görüntü çok karanlık mı?
            if light_level < 50:
                cv2.putText(frame, "Too Dark", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Görüntü çok parlak mı?
            if light_level > 120:
                cv2.putText(frame, "Too Shiny", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Yüz bölgesini çerçeve içine al
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
            # Yüz bölgesi dışındaki pikselleri siyahlaştır
            mask = cv2.rectangle(frame.copy(), (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            mask[y:y+height, x:x+width] = frame[y:y+height, x:x+width]
            
            # Arka planı çıkar
            result = cv2.bitwise_and(frame, mask)
            # Işık miktarını ekranda göster
            cv2.putText(frame, f'Light Level: {light_level:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # İşlenmiş kareyi ekranda göster
    cv2.imshow('Frame', result)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera akışını durdur ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()




