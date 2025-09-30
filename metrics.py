import torch
import torch.nn.functional as F
from math import log10
import piq  # 🔥 better perceptual metrics (pip install piq)

# PSNR
def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 20 * log10(1.0 / torch.sqrt(mse))

# SSIM
def ssim(pred, target):
    return piq.ssim(pred, target, data_range=1.0).item()

# MSE
def mse(pred, target):
    return F.mse_loss(pred, target).item()
