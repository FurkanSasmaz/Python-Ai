# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:32:25 2024

@author: furkan.sasmaz
"""
# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import cv2

def calculate_brightness(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the mean brightness value
    mean_brightness = np.mean(gray)
    return mean_brightness



def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# load the original image
original = cv2.imread(args["image"])

# calculate mean brightness for the original image
original_brightness = calculate_brightness(original)
print("Original Image Brightness:", original_brightness)

for gamma in np.arange(0.0, 3.5, 0.5):
    # ignore when gamma is 1 (there will be no change to the image)
    if gamma == 1:
        continue

    # apply gamma correction
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)

    # calculate mean brightness for the adjusted image
    mean_brightness = cv2.mean(adjusted)[0]
    
    # display the images and mean brightness
    text = "g={}, brightness={:.2f}".format(gamma, mean_brightness)
    cv2.putText(adjusted, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    
    # show the images using cv2.imshow
    cv2.imshow("Original", original)
    cv2.imshow("Gama&Brightness", adjusted)
    
    cv2.waitKey(0)

cv2.destroyAllWindows()

    