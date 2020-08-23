import cv2
import numpy as np
import os
from glob import glob
import sys
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
import argparse
import re
import json

from natsort import natsorted, ns

sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.2)

from time import time

from image_metrics import Metric

from models import *
from train_utils import *

def findLargestContour(mask_img):
    contours, _ = cv2.findContours(mask_img, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bestArea = 200 # Minimum area
    bestContour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > bestArea:
            bestContour = c
            bestArea = area

    assert bestContour is not None
    return bestContour, bestArea

def blobAreaPercentage(orig_raw, recons_raw):
    ''' Given the original raw image and the reconstructed raw image, will 
    compute the ratio of the area between the largest blob in both images.

    '''

    # Load same config used to generate the masks dataset
    config = loadConfig("config2.json")

    # Generate mask using the color detection and extract largest contour
    # from original image
    orig_hsv = cv2.cvtColor(orig_raw, cv2.COLOR_RGB2HSV)
    orig_mask = np.zeros(orig_hsv.shape[:2], dtype=np.uint8)
    for color_range in config['color_ranges']:
        lower_range = np.array(color_range['lower'])
        upper_range = np.array(color_range['upper'])
        currMask = cv2.inRange(orig_hsv, lower_range, upper_range)
        orig_mask = cv2.bitwise_or(orig_mask, currMask)
    
    orig_contour, orig_area = findLargestContour(orig_mask)

    # Ditto for the reconstructed raw
    recons_hsv = cv2.cvtColor(recons_raw, cv2.COLOR_RGB2HSV)
    recons_mask = np.zeros(recons_hsv.shape[:2], dtype=np.uint8)
    for color_range in config['color_ranges']:
        lower_range = np.array(color_range['lower'])
        upper_range = np.array(color_range['upper'])
        currMask = cv2.inRange(recons_hsv, lower_range, upper_range)
        recons_mask = cv2.bitwise_or(recons_mask, currMask)

    recons_contour, recons_area = findLargestContour(recons_mask)

    ratio = recons_area / orig_area

    #print(f"Original area: {orig_area:.2f}, Recons Area: {recons_area:.2f} | Ratio: {ratio:.2f}")

    return ratio

def getAllVideoFrames(videoFn, resizeShape=(640, 480)):
    videoCap = cv2.VideoCapture(videoFn)
    frames = []
    
    _, frame = videoCap.read()
    while frame is not None:
        frame = cv2.resize(frame, resizeShape)
        frames.append(frame)
        _, frame = videoCap.read()

    return np.array(frames).astype(np.float32)

class VideoGenerator:
    def __init__(self, videoFn, resizeShape, batch):
        self.cap = cv2.VideoCapture(videoFn)
        self.batch = batch
        self.resizeShape = resizeShape

    def getFrames(self, rgb=True):
        frames = []
        _, frame = self.cap.read()
        while frame is not None:
            frame = cv2.resize(frame, self.resizeShape)
            # Model was trained with images opened with PIL, so channels must 
            # be in RGB order
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)
            if len(frames) >= self.batch:
                yield np.array(frames).astype(np.uint8)
                frames = []

            _, frame = self.cap.read()

        if len(frames) > 0:
            yield np.array(frames).astype(np.uint8)
        else:
            return 

def normToTensor(imgs):
    # 0~255 -> 0~1
    imgs = np.divide(imgs, 255.)
    # B, H, W, C -> B, C, H, W
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    # Return torch tensor
    return torch.from_numpy(imgs).float().cuda()

def unnorm(imgs):
    imgs = np.multiply(imgs, 255.)
    imgs = np.clip(imgs, 0, 255)
    return imgs.astype(np.uint8)

def plotMeasurements(titleName, metric, metric_name, save_path, show=False):
    fig = plt.figure(figsize=(16, 10))
    plt.title(titleName)

    firstSample = metric.data[0]
    fields = firstSample._fields

    history = np.array(metric.data)
    legends = []

    for i, f in enumerate(fields):
        average = np.average(history[:, i])
        rolling_avg_window = deque(maxlen=30)
        rolling_avg = []
        # Compute rolling average over all data points
        for data in history[:, i]:
            rolling_avg_window.append(data)
            rolling_avg.append(np.average(rolling_avg_window))

        #print(history[:, i].shape, len(rolling_avg))
        plt.plot(history[:, i], linewidth=1)
        plt.plot(rolling_avg, linestyle=':',
                linewidth=3)
        plt.plot(np.array([average]*len(rolling_avg)), 
                linewidth=1.5,
                linestyle="dotted",
                color="red")
        legends.append(f"{metric_name}")
        legends.append("Rolling Avg. (30 frames)")
        legends.append(f"Average = {average:.3f}")
        plt.xlabel("Frame Index")

    plt.legend(legends)

    if show:
        plt.show()

    figFn = titleName + '.jpg'
    path = os.path.join(save_path, figFn)
    print(f"Saving `{path}`.")
    plt.savefig(path)
    plt.close()

def runMetricsOnData(video_fns, model_filename, experiment_folder,
        filters, checkpoint_mode, interval, batch_size):

    metrics_save_path = os.path.join(experiment_folder, 'video_metrics')
    if not os.path.exists(metrics_save_path):
        os.mkdir(metrics_save_path)

    chkpt_folder = os.path.join(experiment_folder, 'chkpt')
    # Retrieve checkpoints filenames
    chkpt_fns = os.listdir(chkpt_folder)
    # Uses natural sort, assuming files end with the epoch number
    chkpt_fns = natsorted(chkpt_fns)

    if checkpoint_mode == 'latest':
        selected_chkpts = [chkpt_fns[-1]]
    elif checkpoint_mode == 'all':
        selected_chkpts = chkpt_fns
    elif checkpoint_mode == 'interval':
        selected_chkpts = [chkpt_fns[i] for i in 
                range(0, len(chkpt_fns), interval)]

    # When the model is running for the first time it will print out what kind
    # model the output seems to be (either Raw+Mask or Raw-only model). This
    # flag is used to make sure that the print only happens once
    INFO_TYPE_FLAG = False

    # Not super important just used for printing and titling the graphs
    # Assuming binary stochastization model, and that the dimensions of the
    # bottleneck output is NFilters x 8 x 10
    bits_per_frame = filters * 10 
    print(f"Will load model `{model_filename}` with `{bits_per_frame}` bits per frame.")

    for curr_chkpt in selected_chkpts:
        print(f"Loading `{curr_chkpt}`.")
        experiment_epoch_number = re.findall(r'\d+', curr_chkpt)[-1]

        # Load specified model and latest checkpoint
        model = eval(model_filename).Net(bottleneckFilters=filters)
        curr_chkpt_path = os.path.join(chkpt_folder, curr_chkpt)
        model.load_state_dict(torch.load(curr_chkpt_path))
        model.eval()
        model.cuda()

        video_metrics = []
        ratio_history = []
        for video_fn in video_fns:
            # Extracting video filename, using for titiling the graph

            video_name = video_fn.split('/')[-1]
            # Create generator for the video
            framesGenerator = VideoGenerator(video_fn, 
                        resizeShape=(320, 240), batch=batch_size)

            video_metrics = Metric("image_metrics/config.json")

            batchCount = 0
            for batchOriginal in framesGenerator.getFrames():
                #print(f"Batch {batchCount+1} for video `{video_fn}`.")
                # Normalize values to 0~255
                batchNorm = normToTensor(batchOriginal)

                # Get reconstructed frames from the model
                out = model(batchNorm)
                if type(out) == tuple:
                    if not INFO_TYPE_FLAG:
                        INFO_TYPE_FLAG = True
                        print("*** This seems to be a Raw+Mask model.")
                    out = out[0]
                elif not INFO_TYPE_FLAG:
                    INFO_TYPE_FLAG = True
                    print("*** This seems to be a Raw only model.")

                # Unnormalize
                reconsUnnorm = tensorToCV_RGB(out)

                ratio = blobAreaPercentage(batchOriginal[0], reconsUnnorm[0])
                ratio_history.append(ratio)
                #input("Done bro.")

                # Compute metrics, comparing reconstructed frames with original 
                # frames
                video_metrics.run(batchOriginal, reconsUnnorm)
                batchCount += 1

            # Plot the metrics for the current video
            for metric_name, metric in video_metrics.metrics.items():
                title = f"{metric_name} on \"{video_name}\" - {bits_per_frame} bits per frame - Epoch {experiment_epoch_number}"
                plotMeasurements(title, metric, metric_name, metrics_save_path)
        avg_ratio = np.mean(ratio_history)
        std_ratio = np.std(ratio_history)
        with open(os.path.join(metrics_save_path, "detectionAreaRatio.log"), 'w') as f:
            f.write(f"Average ratio: {avg_ratio:.3f} | Std deviation: {std_ratio:.3f}\n")

if __name__ == "__main__":
    cm_modes = ['latest', 'all', 'interval']
    parser = argparse.ArgumentParser(description="Run metrics for Mask Autoencoder.")
    parser.add_argument('model_filename',
            help="Filename of the model.")
    parser.add_argument('filters',
            type=int,
            help="Number of filters specified in the model.")
    parser.add_argument('experiment_folder',
            help="Points to the experiment folder, containing the trained "
                 "checkpoints and where the graphs will be saved.")
    parser.add_argument('video_path',
            help="Path to the video used to compute the metrics.")
    parser.add_argument('-bs', '--batch-size', type=int, default=4,
            help="Batch size used when processing the video.")
    parser.add_argument('-cm', '--checkpoint-mode', default='latest',
            help=f"Defines which mode of checkpoint to use. `{cm_modes}`. If mode is `interval` specify the interval with -i")
    parser.add_argument('-i', '--interval', type=int,
            help="Checkpoint files interval.")
    args = parser.parse_args()

    print(args.checkpoint_mode)
    if args.checkpoint_mode not in cm_modes:
        print(f"Invalid checkpoint mode. Should be one of: `{cm_modes}`.")
        sys.exit(1)
    if args.checkpoint_mode == 'interval' and args.interval == None:
        print("Checkpoint mode is `interval`, please specify it with -i.")
        sys.exit(1)

    # TODO: Actually specify a list with the argparser 
    video_fns = [args.video_path]

    startTime = time()
    runMetricsOnData(video_fns, 
            args.model_filename, 
            args.experiment_folder,
            args.filters, 
            batch_size=args.batch_size,
            checkpoint_mode=args.checkpoint_mode,
            interval=args.interval)
    endTime = time()

    elapsed = endTime - startTime
    print(f"Took {elapsed:.2f} seconds.")
