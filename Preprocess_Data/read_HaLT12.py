import numpy as npy
import scipy
import scipy.io
import os
import numpy as np
import torch

data_folder = '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/HaLT12'
data_all = {}
subject_map = {'A':1,
                'B':2,
                'C':3,
                'E':4,
                'F':5,
                'G':6,
                'H':7,
                'I':8,
                'J':9,
                'K':10,
                'L':11,
                'M':12
              }
for i in range(12):
  data_all[i + 1] = []
for f in os.listdir(data_folder):
    subject_id = subject_map[f[11]]
    print(subject_id)
    file_name = os.path.join(data_folder, f)
    mat_file = scipy.io.loadmat(file_name)['o'][0][0]
    marker = mat_file[4]
    data = mat_file[5]
    action_step = np.where(np.abs(np.diff(marker.reshape(marker.shape[0]))) > 0)[0] + 1
    for i in range(0, len(action_step), 2):
        data_all[subject_id].append([torch.tensor(data[action_step[i] : action_step[i + 1]]), marker[action_step[i] + 10]])
        #print(marker[action_step[i] + 10])

    #print(marker.shape, data.shape)

          
with open(os.path.join('/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/HaLT12', 'HaLT12.npy'), 'wb') as f:
    np.save(f, np.array(data_all, dtype=object))          

