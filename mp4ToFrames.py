import sys
import cv2
import numpy as np
import os
from os import path
import argparse

parser = argparse.ArgumentParser(description=
        "Converts video files to video frames.")
parser.add_argument("videos", help="Path to folder containing the video files.")
parser.add_argument("output_folder", help="Path to output folder.")
parser.add_argument("--frame_skip", type=int, default=1, help="Number of frames to skip.")
args = parser.parse_args()

# Will resize the frames to this resolution
NET_INPUT_WIDTH = 320
NET_INPUT_HEIGHT = 240

trainingVideosPath = path.join(args.videos, "training")
validationVideosPath = path.join(args.videos, "validation")
outputRoot = args.output_folder 
outputTraining = path.join(outputRoot, "training")
outputValidation = path.join(outputRoot, "validation")

if not os.path.exists(trainingVideosPath):
    print(f"Couldn't find path `{trainingVideosPath}` .")
    sys.exit(1)
if not os.path.exists(validationVideosPath):
    print(f"Couldn't find path `{validationVideosPath}` .")
    sys.exit(1)
if not os.path.exists(outputRoot):
    os.mkdir(outputRoot)
if not os.path.exists(outputTraining):
    os.mkdir(outputTraining)
if not os.path.exists(outputValidation):
    os.mkdir(outputValidation)

for folder_name in os.listdir(trainingVideosPath):
    currReadFolder = path.join(trainingVideosPath, folder_name)
    currOutputFolder = path.join(outputTraining, folder_name)
    if not os.path.exists(currOutputFolder):
        os.mkdir(currOutputFolder)

    for video_fn in os.listdir(currReadFolder):
        if not video_fn.endswith(".avi"):
            print(f"Skipping `{video_fn}`, doesn't end with .avi")
            continue
    
        print(f"Processing `{video_fn}`.")
        name = video_fn[:-4]
    
        # Open video capture
        videoCap = cv2.VideoCapture(path.join(currReadFolder, video_fn))
        frameCount = 0
        # Read first frame pre-loop
        ret, frame = videoCap.read()
        while frame is not None:
            output_fn = path.join(currOutputFolder, f"{name}_{frameCount}.jpg")
    
            if frameCount % args.frame_skip == 0:
                frame = cv2.resize(frame, (NET_INPUT_WIDTH, NET_INPUT_HEIGHT))
                cv2.imwrite(output_fn, frame)
    
            frameCount += 1
            ret, frame = videoCap.read()
    
        videoCap.release()

for folder_name in os.listdir(validationVideosPath):
    currReadFolder = path.join(validationVideosPath, folder_name)
    currOutputFolder = path.join(outputValidation, folder_name)
    if not os.path.exists(currOutputFolder):
        os.mkdir(currOutputFolder)

    for video_fn in os.listdir(currReadFolder):
        if not video_fn.endswith(".avi"):
            print(f"Skipping `{video_fn}`, doesn't end with .avi")
            continue
    
        print(f"Processing `{video_fn}`.")
        name = video_fn[:-4]
    
        # Open video capture
        videoCap = cv2.VideoCapture(path.join(currReadFolder, video_fn))
        frameCount = 0
        # Read first frame pre-loop
        ret, frame = videoCap.read()
        while frame is not None:
            output_fn = path.join(currOutputFolder, f"{name}_{frameCount}.jpg")
    
            if frameCount % args.frame_skip == 0:
                frame = cv2.resize(frame, (NET_INPUT_WIDTH, NET_INPUT_HEIGHT))
                cv2.imwrite(output_fn, frame)
    
            frameCount += 1
            ret, frame = videoCap.read()
    
        videoCap.release()
