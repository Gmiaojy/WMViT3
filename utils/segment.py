# -*- coding: utf-8 -*-
# @Author  : Miao Guo
# @University  : ZheJiang University
import cv2
import os
from skimage.measure import label, regionprops
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.util import img_as_uint, img_as_ubyte

imagePath = "wmvit3/datas/ORI/20250611_171249_920.jpg"
outputFolder = "wmvit3/datas/ROI"

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
    print(f'The output folder has been created: {outputFolder}')

try:
    originalImage = imread(imagePath)
except FileNotFoundError:
    raise IOError(f'The image cannot be loaded. Please check the path: {imagePath}')

if originalImage.ndim == 3:
    originalImage = rgb2gray(originalImage)

originalImage_uint16 = img_as_uint(originalImage)   # Save another high-precision image for final cropping.
# imsave(os.path.join(outputFolder, 'original.png'), originalImage_uint16)

originalImage_uint8 = img_as_ubyte(originalImage)  # All image operations are performed on uint8 bitmaps.

# --- step1: Median filter ---
blurredImage = cv2.medianBlur(originalImage_uint8, 15)
# imsave(os.path.join(outputFolder, 'med_blurred.png'), blurredImage)

# --- step2: Adaptive threshold processing ---
binaryImage = cv2.adaptiveThreshold(blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 5, 1.8)
# imsave(os.path.join(outputFolder, 'thred.png'), binaryImage)

# --- step3: Morphological processing ---
seErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
erodedImage = cv2.erode(binaryImage, seErode)
# imsave(os.path.join(outputFolder, 'eroded.png'), erodedImage)

seDilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
dilatedImage = cv2.dilate(erodedImage, seDilate)
# imsave(os.path.join(outputFolder, 'dilated.png'), dilatedImage)

# --- step 4: Draw ROI ---
label_image = label(erodedImage)
stats = regionprops(label_image)

baseFileName = os.path.splitext(os.path.basename(imagePath))[0]
minArea = 1000 # Filter areas smaller than minArea
imageForDrawing = cv2.cvtColor(originalImage_uint8, cv2.COLOR_GRAY2BGR)

saved_region_counter = 1
# Traverse all connected regions
for props in stats:
    if props.area > minArea:
        minr, minc, maxr, maxc = props.bbox
        x, y, w, h = minc, minr, maxc - minc, maxr - minr

        cropConstant = 280
        exp_x = int(x - cropConstant / 2)
        exp_y = int(y - cropConstant / 2)
        exp_w = int(w + cropConstant)
        exp_h = int(h + cropConstant)

        img_h, img_w = originalImage_uint8.shape
        exp_x = max(0, exp_x)
        exp_y = max(0, exp_y)

        cv2.rectangle(imageForDrawing, (exp_x, exp_y), (exp_x + exp_w, exp_y + exp_h), (255, 0, 0), 8)

        cropped_region = originalImage_uint16[exp_y: exp_y + exp_h, exp_x: exp_x + exp_w]

        # crop on the 16 - bit precision image
        outputFileName = os.path.join(outputFolder, f'{baseFileName}_region_{saved_region_counter}.png')
        imsave(outputFileName, cropped_region)  
        print(f'Saved: {outputFileName}')
        
        saved_region_counter += 1

imsave(os.path.join(outputFolder, f'{baseFileName}_ROI.png'), imageForDrawing)

# --- show results  ---
"""
import matplotlib.pyplot as plt

plt.figure(1)
plt.imshow(originalImage_uint16, cmap='gray')
plt.title('Original Image')
plt.show()

plt.figure(2)
plt.imshow(blurredImage, cmap='gray')
plt.title('Blurred Image')
plt.show()

plt.figure(3)
plt.imshow(binaryImage, cmap='gray')
plt.title('Binary Image')
plt.show()

plt.figure(4)
plt.imshow(dilatedImage, cmap='gray')
plt.title('Dilated Image')
plt.show()

plt.figure(5)
plt.imshow(erodedImage, cmap='gray')
plt.title('Eroded Image')
plt.show()

plt.figure(6)
plt.imshow(cv2.cvtColor(imageForDrawing, cv2.COLOR_BGR2RGB))
plt.title('Image with ROI')
plt.show()
"""