import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from models import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
from PIL import Image

def showImgs(img_tensor):
    detransform = transforms.ToPILImage()
    img = img_tensor[0]
    pil_img = detransform(img)

def generateVideo(model_filename, model_checkpoint, video_fn):

    transform = transforms.Compose(
      #[transforms.Resize((240, 320)),
       [transforms.ToTensor()]
    )
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    detransform = transforms.ToPILImage()

    model = eval(model_filename).Net(128)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.cuda()

    cap = cv2.VideoCapture(video_fn)
    output_fn = video_fn.split('/')[-1]
    output_fn = output_fn[:-4] + "_recons.avi"
    out = cv2.VideoWriter(output_fn, 
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (640, 240))

    print(f"Processing {output_fn}.")
    frameCount = 0
    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        frame = cv2.resize(frame, (320, 240))

        # Convert to PIL Image
        pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(pil_frame)

        tensor_frame = transform(pil_frame).unsqueeze(0)
        tensor_frame = tensor_frame.cuda()

        output = model(tensor_frame)
        if type(output) == tuple:
            pred_raws, pred_masks = output
        else:
            pred_raws = output

        output = torch.clamp(pred_raws, 0, 1)

        cv_img = detransform(output.squeeze(0).cpu())
        cv_img = cv2.cvtColor(np.array(cv_img), cv2.COLOR_RGB2BGR)

        cat = np.hstack([frame, cv_img])
        out.write(cat)

        if frameCount % 100 == 0:
            print(f"{frameCount} frames processed.")

        frameCount += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model-filename> <model-checkpoint> <video>")
        sys.exit(1)

    model_filename = sys.argv[1]
    model_checkpoint = sys.argv[2]
    video_fn = sys.argv[3]

    generateVideo(model_filename, model_checkpoint, video_fn)
