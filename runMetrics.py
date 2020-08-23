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
from train_utils import *

from natsort import natsorted, ns

sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.2)

from time import time

from image_metrics import Metric

from models import *
from train_utils import *

def getLargestBlobROICoords(mask):
    ''' Given a mask, will return the coordinates of the largest blob present
    in the image, used for cropping.

    '''

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    bestArea = 100 # Set minimum contour
    bestContour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > bestArea:
            bestContour = c
            bestArea = area 

    assert bestContour is not None

    # Find bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(bestContour)

    # x1, x2, y1, y2
    return x, x+w, y, y+h
    

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

def runMetricsOnData(dataset_path, model_filename, experiment_folder,
        filters, checkpoint_mode, interval, batch_size):

    metrics_save_path = os.path.join(experiment_folder, 'val_metrics')
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

    # Not super important just used for printing and titling the graphs
    # Assuming binary stochastization model, and that the dimensions of the
    # bottleneck output is NFilters x 8 x 10
    bits_per_frame = filters * 10 
    print(f"Will load model `{model_filename}` with `{bits_per_frame}` bits per frame.")

    # Load dataset
    testset = ImageFolderMask(root_path=dataset_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)

    for curr_chkpt in selected_chkpts:
        print(f"Loading `{curr_chkpt}`.")
        experiment_epoch_number = re.findall(r'\d+', curr_chkpt)[-1]

        # Load specified model and latest checkpoint
        model = eval(model_filename).Net(bottleneckFilters=filters)
        curr_chkpt_path = os.path.join(chkpt_folder, curr_chkpt)
        model.load_state_dict(torch.load(curr_chkpt_path))
        model.eval()
        model.cuda()

        # Creates the metric tracker object
        dataset_metrics = Metric("image_metrics/config.json")
        roi_metrics = Metric("image_metrics/config.json")

        batchCount = 0
        for i, data in enumerate(testloader, 0):
            orig_raws, orig_masks = data[0].cuda(), data[1].cuda()

            # Get reconstructed frames from the model
            out = model(orig_raws)
            if type(out) == tuple:
                pred_raws, pred_masks = out
            else:
                pred_raws = out

            # Convert tensors to OpenCV images
            rgb_orig_masks = tensorToCV_RGB(orig_masks)
            rgb_orig_raws = tensorToCV_RGB(orig_raws)
            #rgb_pred_masks = tensorToCV_RGB(pred_masks)
            rgb_pred_raws = tensorToCV_RGB(pred_raws)

            # Extract ROIs from the original and reconstructed images, based
            # on the original masks
            orig_rois = []
            pred_rois = []
            i = 0
            for orig_raw, orig_mask, pred_raw in zip(rgb_orig_raws, 
                    rgb_orig_masks, rgb_pred_raws):

                x1, x2, y1, y2 = getLargestBlobROICoords(orig_mask)
                if x2 - x1 <= 20 or y2 - y1 <= 20:
                    print("skip")
                    continue

                orig_rois.append(orig_raw[y1:y2, x1:x2])
                pred_rois.append(pred_raw[y1:y2, x1:x2])
                cv2.imwrite(f"orig_{i}.jpg", orig_rois[i])
                cv2.imwrite(f"pred_{i}.jpg", pred_rois[i])
                i += 1

            # Compute metrics, comparing reconstructed frames with original 
            # frames
            dataset_metrics.run(rgb_orig_raws, rgb_pred_raws)
            roi_metrics.run(orig_rois, pred_rois)
            batchCount += 1

        # Plot the metrics for the current video
        for metric_name, metric in dataset_metrics.metrics.items():
            title = f"{metric_name} - {bits_per_frame} bits per frame - Epoch {experiment_epoch_number}"
            plotMeasurements(title, metric, metric_name, metrics_save_path)

        for metric_name, metric in roi_metrics.metrics.items():
            title = f"ROI ONLY - {metric_name} - {bits_per_frame} bits per frame - Epoch {experiment_epoch_number}"
            plotMeasurements(title, metric, metric_name, metrics_save_path)

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
    parser.add_argument('dataset_path',
            help="Path to the root of dataset used to compute the metrics.")
    parser.add_argument('-bs', '--batch-size', type=int, default=4,
            help="Batch size used when processing the dataset.")
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

    startTime = time()
    runMetricsOnData(args.dataset_path, 
            args.model_filename, 
            args.experiment_folder,
            args.filters, 
            batch_size=args.batch_size,
            checkpoint_mode=args.checkpoint_mode,
            interval=args.interval)
    endTime = time()

    elapsed = endTime - startTime
    print(f"Took {elapsed:.2f} seconds.")
