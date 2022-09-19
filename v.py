#Import numpy
import numpy as np

#Import scikitlearn for machine learning functionalities
import sklearn
from sklearn.manifold import TSNE 
from sklearn.datasets import load_digits # For the UCI ML handwritten digits dataset

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch
import seaborn as sb

import torchaudio
import torchaudio.transforms as T

from typing import Tuple
import copy
import time

from torch.utils.data import dataset
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import librosa
import argparse
import scipy
import os

from Baselines.MAE import *
from Baselines.SimCLR import *
from Baselines.CLOCS import *
from Baselines.NewCLOCS import *
from Baselines.BYOL import *
from Baselines.MocoV3 import *
from Backbone.Transformer_Backbone import *
from dataset import *
from train import *
from augmentation import *
from supervised import *

def plot(x, colors, n_classes):
    palette = np.array(sb.color_palette("hls", n_classes))  #Choosing color palette 
    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    
    sc = ax.scatter(x[:,0], x[:,1], c=palette[colors.astype(np.int32)])# , lw=0
    for i in range(x.shape[0]):
      print(x[i], colors[i])

    txts = []
    for i in range(0, n_classes):
        xtext, ytext = np.mean(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=10)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    return f, ax, txts


"""
dataset_path = '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motor_Imagery_9'
X = []
Y = []
ids = []
split_id = [1]
data_class = ['data1', 'data2', 'data3', 'data4']
        
file_num_range = ['A0' + str(i) + 'T.mat'for i in split_id]
file_num_range.extend(['A0' + str(i) + 'E.mat'for i in split_id])

for set_num in file_num_range:
    file_name = set_num
    subject_id = int(file_name[2])
    mat = scipy.io.loadmat(os.path.join(dataset_path, file_name))
    for i in range(4): # each file has 4 classes
        data_num = mat[data_class[i]].shape[2]
        for j in range(data_num): # each class has a specific number of data
            X.append(mat[data_class[i]][:,:,j])
            Y.append(i)
            ids.append(subject_id)
X = torch.tensor(np.array(X)).float()
Y = np.array(Y)

max_eeg = X.mean(axis = (1, 2), keepdims = True)
min_eeg = X.std(axis = (1, 2), keepdims = True)
X = (X - max_eeg)/min_eeg

print(X.shape)"""
"""
data = np.load('/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EMG_Ninapro/emg_20.npy', allow_pickle = True).item()
subjects = [14]#[9, 2, 14, 13]#[11, 19, 17, 15, 1, 18, 12, 3, 4, 10, 6, 8]
X = []
Y = []

for subject_id in subjects:
    X.extend([np.array(data[subject_id]['emg'][i][400:400+9000]) for i in range(len(data[subject_id]['emg']))])
    Y.extend([data[subject_id]['label'][i] for i in range(len(data[subject_id]['label']))])
X = np.stack(X, axis = 0)
Y = np.array(Y)

max_eeg = X.mean(axis = (1, 2), keepdims = True)
min_eeg = X.std(axis = (1, 2), keepdims = True)
X = (X - max_eeg)/min_eeg
"""

data = np.load('/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109.npy', allow_pickle = True).item()
subjects = [24, 36, 94, 65, 45, 9, 17, 77, 12, 14, 69, 50, 95, 99, 61, 46, 3, 96, 30, 82, 25, 108, 59, 71, 11, 31, 4, 23, 56, 90, 20, 7, 101, 85, 49, 38, 102, 29, 35, 83, 109, 15, 91, 75, 5, 60, 21, 84, 16, 22, 32, 86, 1, 72, 78, 52, 54, 74, 26, 107, 48, 106, 2, 8, 105, 68, 58, 98, 27, 41, 57, 97, 80, 73, 62, 44, 10, 87, 40, 19, 79, 70, 33, 81, 13, 93, 37, 18, 66, 28, 76]#, 36, 94, 65, 45, 9, 42, 17, 77
#[47, 63, 39, 53, 64, 67, 34, 6, 55, 103, 51]
X = []
Y = []
for subject_id in subjects:
    for i in range(len(data[subject_id])):
        if data[subject_id][i][1] == 0:
          continue
        Y.append(data[subject_id][i][1] - 1)
        print(len(data[subject_id][i][0]))
        plt.plot(data[subject_id][i][0][0])
  
        if len(data[subject_id][i][0]) >= 646:
            X.append(np.array(data[subject_id][i][0][:646]))
        else:
            X.append(np.pad(data[subject_id][i][0], ((0, 646 - data[subject_id][i][0].shape[0]), (0, 0)), 'constant', constant_values=0))
X = np.stack(X, axis = 0)
Y = np.array(Y)

max_eeg = X.mean(axis = (1, 2), keepdims = True)
min_eeg = X.std(axis = (1, 2), keepdims = True)
X = (X - max_eeg)/min_eeg

file_path = '/content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/EEG109_CNN_1.0Pretrained_False.pt'
model = torch.load(file_path).cuda()
with torch.no_grad():
    if 'Backbone' in file_path:
        print('backbone model!')
        X = model(torch.tensor(X).float().cuda().transpose(1, 2))
    else:
        X = model.encoder(torch.tensor(X).float().cuda().transpose(1, 2))#.encoder
digits_final = TSNE(perplexity=100, init='pca').fit_transform(X.squeeze(-1).detach().cpu().numpy()) 
#digits_final = sklearn.decomposition.PCA(n_components = 2).fit_transform(X.squeeze(-1).detach().cpu().numpy()) 
plot(digits_final,Y, 4)
plt.savefig('r.png')


