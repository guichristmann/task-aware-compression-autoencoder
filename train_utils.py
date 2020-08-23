import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os

def loadConfig(fn):
    with open(fn) as f:
        config = json.load(f)

    return config

def showImgs(img_tensor):
    detransform = transforms.ToPILImage()
    img = img_tensor[0]
    pil_img = detransform(img)

def tensorToCV_RGB(imgs_tensor):
    # Copy tensor to cpu
    imgs_tensor = imgs_tensor.cpu()
    # Create PIL detransform
    detransform = transforms.ToPILImage()

    # Iterate image tensors, converting to CV images
    cv_imgs = []
    for tensor_img in imgs_tensor: 
        cv_rgb_img = np.array(detransform(tensor_img))
        cv_imgs.append(cv_rgb_img)

    return cv_imgs

def rgbToColorMask(rgb_imgs, config):
    # Retrieve color ranges from config
    lower_range = np.array(config["color_lower_range"])
    upper_range = np.array(config["color_upper_range"])

    # Create masks
    masks = []
    for rgb_img in rgb_imgs:
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, lower_range, upper_range)
        masks.append(np.expand_dims(mask, axis=0))

    # Normalize
    masks = np.array(masks).astype(np.float32)
    masks = np.divide(masks, 255.)

    return masks 

def plotHistory(history, savepath):
    fig = plt.figure(figsize=(16, 10))

    raw_loss = history["raw_loss"]
    loss = history["loss"]
    mask_loss = history["mask_loss"]
    epochs = len(raw_loss)

    plt.plot(np.arange(epochs, dtype=np.int32), loss)
    plt.plot(np.arange(epochs, dtype=np.int32), raw_loss)
    plt.plot(np.arange(epochs, dtype=np.int32), mask_loss)

    plt.legend(["Weighted Loss", "Raw Loss", "Mask Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    filename = f"Graph_{epochs}.jpg"
    path = os.path.join(savepath, filename)
    plt.savefig(path)
    plt.close()

class ImageFolderMask(Dataset):
    def __init__(self, root_path, prefix="mask"):
        self.raw_files = []
        self.mask_files = []

        self.prefix = prefix

        for folder in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder)
            for fn in os.listdir(folder_path):
                fullPath = os.path.join(folder_path, fn)
                if fn.startswith(prefix):
                    self.mask_files.append(fullPath)
                else:
                    self.raw_files.append(fullPath)

        assert len(self.raw_files) == len(self.mask_files)
        self.raw_files = sorted(self.raw_files)
        self.mask_files = sorted(self.mask_files)

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, index):
        raw_path = self.raw_files[index]
        mask_path = self.mask_files[index]

        # Loading images as arrays with PIL
        raw_img = np.array(Image.open(raw_path))
        mask_img = np.expand_dims(np.array(Image.open(mask_path)), axis=-1)

        # Normalize
        raw_img = np.divide(raw_img, 255.0)
        mask_img = np.divide(mask_img, 255.0)

        # H, W, C -> C, H, W
        raw_img = np.transpose(raw_img, (2, 0, 1))
        mask_img = np.transpose(mask_img, (2, 0, 1))

        # Converting to torch tensors
        raw_tensor = torch.from_numpy(raw_img).float()
        mask_tensor = torch.from_numpy(mask_img).float()

        return raw_tensor, mask_tensor
