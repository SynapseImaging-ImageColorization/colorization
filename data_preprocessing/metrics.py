import math

import cv2
import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, _) = ssim(img1, img2, full=True)

    return score

def calculate_fid(img1, img2):
    img1 = img1.reshape((img1.shape[0], -1))
    img2 = img2.reshape((img2.shape[0], -1))
    mu1, sigma1 = img1.mean(axis=0), np.cov(img1, rowvar=False)
    mu2, sigma2 = img2.mean(axis=0), np.cov(img2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
