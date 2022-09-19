import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

def gaussian_noise(data, sigma):
    noise_data = data + (sigma ** 0.5) * torch.randn(data.shape).cuda()
    return noise_data

def flip(data, dims):
    flip_data = torch.flip(data, dims = dims)
    return flip_data

def stretch_spec(data, device, rate = 0.8):
    """https://pytorch.org/audio/stable/transforms.html"""
    length = data.shape[0]
    data = data.transpose(0, 1)
    # data shape = [hidden dimension, seq length]
    spectrogram = T.Spectrogram(power = None).to(device)
    stretch = T.TimeStretch().to(device)
    inverse_spectrogram = T.InverseSpectrogram().to(device)

    spec = spectrogram(data)
    spec = stretch(spec, rate)    
    waveform = inverse_spectrogram(spec, length)
    return waveform

def mask_dim(data, dim_mask_param, dim = 2):
    l = data.shape[dim] 
    start = torch.randint( high = l - dim_mask_param, size = [1]).item()
    if dim == 1:
      data[:, start: start + dim_mask_param] = 0
    if dim == 0:
      data[start: start + dim_mask_param] =  0
    if dim == 2:
      data[:, :, start: start + dim_mask_param] = 0
    return data

def mask_freq_spec(data, device, freq_mask_param):
    length = data.shape[0]
    data = data.transpose(0, 1)
    # data shape = [hidden dimension, seq length]
    spectrogram = T.Spectrogram(power = None).to(device)
    inverse_spectrogram = T.InverseSpectrogram().to(device)

    spec = spectrogram(data)
    spec = mask_dim(spec, dim_mask_param = int(spec.shape[1] * freq_mask_param), dim = 1)
    waveform = inverse_spectrogram(spec, length)
    return waveform

def mask_time_spec(data, device, time_mask_param):
    length = data.shape[0]
    data = data.transpose(0, 1)
    # data shape = [hidden dimension, seq length]
    spectrogram = T.Spectrogram(power = None).to(device)
    inverse_spectrogram = T.InverseSpectrogram().to(device)

    spec = spectrogram(data)  
    spec = mask_dim(spec, dim_mask_param = int(spec.shape[2] * time_mask_param), dim = 2)
    waveform = inverse_spectrogram(spec, length)
    return waveform
