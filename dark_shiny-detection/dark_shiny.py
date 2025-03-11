# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:29:13 2023

@author: furkan.sasmaz
"""

import cv2

# Kamera kaynağı veya video dosyası
cap = cv2.VideoCapture(0)

while True:
    # Kameradan görüntü yakala
    ret, frame = cap.read()

    # Görüntüyü gri tona dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Görüntünün parlaklığını hesapla
    brightness = cv2.mean(gray)[0]

    # Görüntü çok karanlık mı?
    if brightness < 50:
        cv2.putText(frame, "Too Dark", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Görüntü çok parlak mı?
    if brightness > 120:
        cv2.putText(frame, "Too Shiny", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(brightness)
    # Görüntüyü ekranda göster
    cv2.imshow("frame", frame)

    # Q tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()