import json
import numpy as np
import cv2
import os
import sys
from train_utils import *

def generateColorMaskImages(fnsPath, config):
    total = len(fnsPath)
    count = 0
    for fn in fnsPath:
        # Load image with opencv
        bgr_img = cv2.imread(fn)
        # Convert to HSV and get color mask
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        
        for color_range in config['color_ranges']:
            lower_range = np.array(color_range['lower'])
            upper_range = np.array(color_range['upper'])
            currMask = cv2.inRange(hsv_img, lower_range, upper_range)
            mask = cv2.bitwise_or(mask, currMask)

        # Process new filename
        brokenPath = fn.split('/')
        newFilename = 'mask_' + brokenPath[-1]
        pathToDir = '/'.join(brokenPath[:-1])
        newFilePath = os.path.join(pathToDir, newFilename)
        cv2.imwrite(newFilePath, mask)

        count += 1
        if count % 100 == 99:
            print(f"Processed {count} out of {total} images.")

    print("Finished.")

def getPathToImages(rootPath):
    trainingPath = os.path.join(rootPath, 'training')
    validationPath = os.path.join(rootPath, 'validation')

    if not os.path.exists(trainingPath):
        raise FileNotFoundError(f"Expected `{trainingPath}`.")
    if not os.path.exists(validationPath):
        raise FileNotFoundError(f"Expected `{validationPath}`.")

    # Iterate over all subfolders both in training and validation and append 
    # the filenames
    filesPath = []
    for classFolder in os.listdir(trainingPath):
        classPath = os.path.join(trainingPath, classFolder)
        for fn in os.listdir(classPath):
            if not fn.startswith('mask_'): # Ignore mask images
                fullPath = os.path.join(classPath, fn)
                filesPath.append(fullPath)
    for classFolder in os.listdir(validationPath):
        classPath = os.path.join(validationPath, classFolder)
        for fn in os.listdir(classPath):
            if not fn.startswith('mask_'):
                fullPath = os.path.join(classPath, fn)
                filesPath.append(fullPath)

    return filesPath

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset>")
        sys.exit(1)

    config = loadConfig('config.json')
    rootPath = sys.argv[1]

    # Get images
    fnsPath = getPathToImages(rootPath)
    print(f"Found {len(fnsPath)} images.")
    generateColorMaskImages(fnsPath, config)
