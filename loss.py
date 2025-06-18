from torch.fft import fft2
from torch.nn.functional import l1_loss,relu
from utils import calBacknoise
import torch

def posloss(xguess):

    return torch.sum(relu(-xguess))

def fftloss(x,y):
    conv_x = fft2(x)
    conv_y = fft2(y)
    loss = l1_loss(conv_x,conv_y)
    return loss

def deCrosstalk_loss(centerview,xguess,psf_energy_mean):
    if xguess.ndim == 4: xguess = torch.squeeze(xguess,0)
    xguess_backnoise = calBacknoise(centerview)/psf_energy_mean
    loss = torch.mean(relu(xguess_backnoise-xguess))
    return loss

def zloss(xguess):
    if xguess.ndim ==3: zloss = torch.mean(torch.abs(2*xguess[1:-1,...]-xguess[:-2,:,:]-xguess[2:,:,:]))
    else: zloss = torch.mean(torch.abs(2*xguess[:,1:-1,...]-xguess[:,:-2,:,:]-xguess[:,2:,:,:]))
    return zloss