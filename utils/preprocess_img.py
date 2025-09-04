# -*- coding: utf-8 -*-
# @Author  : Miao Guo
# @University  : ZheJiang University
import math
import os
import argparse
import cv2
import numpy as np
from PIL import Image

def findMax(imgPath):
    """Find the maximum dimension size"""
    maxPixel = 0
    maxPixel_imgPath = None

    for mainPath, dirs, file in os.walk(imgPath, topdown=False):
        for imgName in file:  # read subfolder
            imgpath = os.path.join(mainPath, imgName)
            img = cv2.imread(imgpath)
            try:
                tempPixel = max(img.shape)
                if tempPixel > maxPixel:
                    maxPixel = tempPixel
                    maxPixel_imgPath = imgpath
            except Exception as e:
                print(f"An error occurred while processing the image: {imgpath}, Error: {e}")
    return maxPixel, maxPixel_imgPath


def imgResize(maxPixel,img, img_size):
    """Specify a square black background with the maximum dimension size,
    and scale the image proportionally within the background range."""

    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert to grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("The image size is invalid. The width or height is 0.")

    background = np.zeros((maxPixel, maxPixel), dtype=np.uint8)
    x_offset = (maxPixel - w) // 2
    y_offset = (maxPixel - h) // 2
    background[y_offset:y_offset + h, x_offset:x_offset + w] = img
    resized_background = cv2.resize(background, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized_background


def normalize_and_2unit8(inp):
    """Normalize the input array to the range [0, 255] and convert to uint8."""
    if np.min(inp) == np.max(inp):
        return np.zeros_like(inp,dtype=np.uint8)
    normalized = inp - np.min(inp) 
    stoke = (normalized / np.max(normalized)) * 255  
    img = np.clip(stoke,0,255).astype(np.uint8)
    return img


def CalStokes(args):
    """Calculate the Stokes parameters S0, S1, S2 from the input images."""
    src_path = args.src_path
    det_path = args.det_path 
    img_size = 256  # The size of the finally generated image

    maxPixel, maxPixel_imgPath  = findMax(src_path)
    print(f"maxPixel: {maxPixel}, imgPath: {maxPixel_imgPath}")
    maxPixel = math.ceil(maxPixel / 2)

    for mainPath, dirs, file in os.walk(src_path, topdown=False):
        for imgName in file:  # read subfolder
            imgpath = os.path.join(mainPath, imgName)
            img = Image.open(imgpath)  # read pictures, PIL type
            img = np.array(img)  # (h,w,c)
            h, w = img.shape[:2]
            if w % 2 != 0:
                img = img[:, :-1]
            if h % 2 != 0:
                img = img[:-1, :]

            I_90 = img[::2, ::2].astype(np.float64)  # Polarization angle 90°
            I_45 = img[::2, 1::2].astype(np.float64)  # Polarization angle 45°
            I_135 = img[1::2, ::2].astype(np.float64)  # Polarization angle 135°
            I_0 = img[1::2, 1::2].astype(np.float64)  # Polarization angle 0°

            S0 = I_0 + I_90  # Total luminous intensity
            S1 = I_0 - I_90  # Horizontal and vertical light intensity difference
            S2 = I_45 - I_135  # Diagonal light intensity difference
            L1 = S1 / S0
            L2 = S2 / S0

            # Normalization
            S0_img = normalize_and_2unit8(S0)
            L1_img = normalize_and_2unit8(L1)
            L2_img = normalize_and_2unit8(L2)

            # Process the picture to the specified size
            S0_img = imgResize(maxPixel, S0_img, img_size)
            L1_img = imgResize(maxPixel, L1_img, img_size)
            L2_img = imgResize(maxPixel, L2_img, img_size)

            # Create the storage directory
            relative_path = os.path.relpath(mainPath, src_path)
            det_subdir = os.path.join(det_path, relative_path)
            os.makedirs(det_subdir, exist_ok=True)
            for img, suffix in zip([S0_img, L1_img, L2_img], ['_0', '_1', '_2']):
                cv2.imwrite(os.path.join(det_subdir, imgName[:-4] + f"{suffix}.jpg"), img)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Stokes parameters from polarization images.")
    parser.add_argument('--src_path', type=str, default="wmvit3/datas/ROI", help='Path to the source images.')
    parser.add_argument('--det_path', type=str, default="wmvit3/datas/input_example", help='Path to save the processed images.')
    args = parser.parse_args()
    CalStokes(args)
    # src_path = "wmvit3/datas/After"
        # det_path = "wmvit3/datas/Input"
    src_path = "wmvit3/datas/ROI"
    det_path = "wmvit3/datas/input_example"