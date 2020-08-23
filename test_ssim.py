from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

img1 = cv2.imread("orig_0.jpg")
img2 = cv2.imread("pred_0.jpg")

print(img1.shape, img2.shape)
v = ssim(img1, img2, data_range=img2.max() - img2.min(), multichannel=True)
print(v)
