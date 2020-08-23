import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from models import *
from train_utils import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
from PIL import Image

def train(model_filename, config_fn='config.json', TAG=''):
    if TAG != '':
        SESSION_NAME = f"tl_{model_filename}" + '-' + TAG
    else:
        SESSION_NAME = f"tl_{model_filename}"

    CONFIG = loadConfig(config_fn)

    DATASET_ROOT = "Datasets/ColorMask10Skip"
    #DATASET_ROOT = "Datasets/Archery10Skip"
    #DATASET_ROOT = "Dataset10Skip"

    BATCH_SIZE = 16
    INFO_INTERVAL = 36 # In batches
    SAVE_INTERVAL = 50 # In epochs

    print(f"Session name: {SESSION_NAME}. Dataset: {DATASET_ROOT}")
    input("Press enter to start.")

    os.makedirs(f"Experiments/{SESSION_NAME}/out", exist_ok=True)
    os.makedirs(f"Experiments/{SESSION_NAME}/chkpt", exist_ok=True)
    os.makedirs(f"Experiments/{SESSION_NAME}/graphs", exist_ok=True)

    print(f"Loading data.")
    transform = transforms.Compose(
      [transforms.ToTensor()])

    trainset = ImageFolderMask(root_path=f'{DATASET_ROOT}/training')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=8)

    testset = ImageFolderMask(root_path=f'{DATASET_ROOT}/validation') 
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=8)

    batchPerEpoch = len(trainloader)
    print(f"[Training]: Total of {len(trainset)} images in {len(trainloader)} batches of size {BATCH_SIZE}")
    print(f"[Validation]: Total of {len(testset)} images in {len(testloader)} batches of size {BATCH_SIZE}")

    print(f"Loading model.")
    model = eval(model_filename).Net(bottleneckFilters=128)
    model.train()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mseLoss = nn.MSELoss()

    history = {"raw_loss": [], "mask_loss": [], "loss": []}
    for epoch in range(251): 
        print(f"### Epoch {epoch+1} ###")

        running_loss = 0.0
        running_mask_loss = 0.0
        running_raw_loss = 0.0
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_raw_loss = 0.0
        epoch_start_t = time()
        batch_start_t = time()
        for i, data in enumerate(trainloader, 0):
            in_raws, in_masks = data[0].cuda(), data[1].cuda()

            optimizer.zero_grad()

            # Get outputs from network, the raw reconstruction and the mask
            pred_raws, pred_masks = model(in_raws)

            raw_loss = mseLoss(in_raws, pred_raws)
            mask_loss = mseLoss(in_masks, pred_masks)
            loss = raw_loss + mask_loss
            loss.backward()
            optimizer.step()

            if epoch % SAVE_INTERVAL == SAVE_INTERVAL-1:
                torch.save(model.state_dict(), f'Experiments/{SESSION_NAME}/chkpt/{model_filename}_{epoch+1}.pth')

            # Accumulate losses
            running_raw_loss += raw_loss.item()
            running_mask_loss += mask_loss.item()
            running_loss += loss.item()
            if i % INFO_INTERVAL == INFO_INTERVAL-1:
                batch_elapsed_t = time() - batch_start_t 
                batch_start_t = time()
                print(f'[{epoch+1}, {i+1}] Loss: {running_loss/INFO_INTERVAL:.4f} | Raw Loss: {running_raw_loss/INFO_INTERVAL:.4f} | Mask Loss: {running_mask_loss/INFO_INTERVAL:.4f} @ {batch_elapsed_t:.2f} seconds')
                # Accumulate epoch loss for later calculation
                epoch_loss += running_loss
                epoch_mask_loss += running_mask_loss
                epoch_raw_loss += running_raw_loss
                running_loss = 0.0
                running_raw_loss = 0.0
                running_mask_loss = 0.0

                # Create preview image
                y_raw = torch.cat((in_raws[0], pred_raws[0]), dim=2).unsqueeze(0)
                y_raw = y_raw.clamp(0, 1)

                y_mask = torch.cat((in_masks[0], pred_masks[0]), dim=2).unsqueeze(0)
                y_mask = y_mask.clamp(0, 1)
                # Hack to make the mask image have 3 channels and allow it to be saved
                y_mask_3ch = torch.zeros_like(y_raw)
                y_mask_3ch[:, 0, :, :] = y_mask
                y_mask_3ch[:, 1, :, :] = y_mask
                y_mask_3ch[:, 2, :, :] = y_mask

                y = torch.cat((y_raw, y_mask_3ch), dim=2)

                save_image(y, f'Experiments/{SESSION_NAME}/out/view_{epoch}_{i}.jpg')         

        # Compute average loss for epoch
        epoch_loss += running_loss
        epoch_mask_loss += running_mask_loss
        epoch_raw_loss += running_raw_loss
        epoch_loss = epoch_loss / batchPerEpoch
        epoch_mask_loss = epoch_mask_loss / batchPerEpoch
        epoch_raw_loss = epoch_raw_loss / batchPerEpoch

        history["loss"].append(epoch_loss)
        history["raw_loss"].append(epoch_raw_loss)
        history["mask_loss"].append(epoch_mask_loss)

        epoch_elapsed_t = time() - epoch_start_t
        print(f'Epoch took {epoch_elapsed_t:.2f} seconds.')
        print(f"Avg. epoch Loss: {epoch_loss:.4f} | Avg. raw loss: {epoch_raw_loss:.4f} | Avg. mask Loss: {epoch_mask_loss:.4f}")
        print('=' * 40)

    print('Finished Training')
    plotHistory(history, f'Experiments/{SESSION_NAME}/graphs/')

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3 :
        print(f"Usage: {sys.argv[0]} <model-filename> <experiment tag>")
        sys.exit(1)

    tag = ""
    model_filename = sys.argv[1]
    if len(sys.argv) == 3:
        tag += sys.argv[2]

    train(model_filename, config_fn='config.json', TAG=tag)
