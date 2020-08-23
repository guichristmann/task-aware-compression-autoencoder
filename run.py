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

def showImgs(img_tensor):
    detransform = transforms.ToPILImage()
    img = img_tensor[0]
    pil_img = detransform(img)

def train(model_filename, TAG=''):
    if TAG != '':
        SESSION_NAME = f"{model_filename}" + '-' + TAG
    else:
        SESSION_NAME = f"{model_filename}"

    #DATASET_ROOT = "Dataset10Skip"
    DATASET_ROOT = "Archery10Skip"

    BATCH_SIZE = 16
    INFO_INTERVAL = 11 # In batches
    SAVE_INTERVAL = 50 # In epochs

    print(f"Session name: {SESSION_NAME}.")
    input("Press enter to start.")

    os.makedirs(f"Experiments/{SESSION_NAME}/out", exist_ok=True)
    os.makedirs(f"Experiments/{SESSION_NAME}/chkpt", exist_ok=True)
    os.makedirs(f"Experiments/{SESSION_NAME}/logs", exist_ok=True)

    print(f"Loading data.")
    transform = transforms.Compose(
      [transforms.ToTensor()])
      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder(root=f'{DATASET_ROOT}/training', 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=8)

    testset = torchvision.datasets.ImageFolder(root=f'{DATASET_ROOT}/validation',
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=8)

    print(f"[Training]: Total of {len(trainset)} images in {len(trainloader)} batches of size {BATCH_SIZE}")
    print(f"[Validation]: Total of {len(testset)} images in {len(testloader)} batches of size {BATCH_SIZE}")

    print(f"Loading model.")
    model = eval(model_filename).Net()
    model.train()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10001): 
        print(f"### Epoch {epoch+1} ###")

        running_loss = 0.0
        epoch_start_t = time()
        batch_start_t = time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(inputs, outputs)
            loss.backward()
            optimizer.step()
            #save
            if epoch % SAVE_INTERVAL == SAVE_INTERVAL-1:
                torch.save(model.state_dict(), f'Experiments/{SESSION_NAME}/chkpt/{model_filename}_{epoch+1}.pth')

           # print statistics
            running_loss += loss.item()
            if i % INFO_INTERVAL == INFO_INTERVAL-1:
                batch_elapsed_t = time() - batch_start_t 
                batch_start_t = time()
                print(f'[{epoch+1}, {i+1}] loss: {running_loss/INFO_INTERVAL:.4f} @ {batch_elapsed_t:.2f} seconds')

                running_loss = 0.0

                y = torch.cat((inputs[0], outputs[0]), dim=2).unsqueeze(0)
                y = y.clamp(0, 1)
                save_image(y, f'Experiments/{SESSION_NAME}/out/img_{epoch}_{i}.jpg')         
    
        epoch_elapsed_t = time() - epoch_start_t
        print(f'Epoch took {epoch_elapsed_t:.2f} seconds')

    print('Finished Training')

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3 :
        print(f"Usage: {sys.argv[0]} <model-filename> <experiment tag>")
        sys.exit(1)

    tag = ""
    model_filename = sys.argv[1]
    if len(sys.argv) == 3:
        tag += sys.argv[2]

    train(model_filename, tag)
