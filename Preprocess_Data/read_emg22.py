import numpy as npy
import scipy
import scipy.io
import os
import numpy as np

data_folder = '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EMG_Ninapro'

data_all = {}

for i in range(1, 21):
    data_all[i] = {}
    data_all[i]['emg'] = []
    data_all[i]['label'] = []
    data_all[i]['ids'] = []
    folder_name = 'Subject_' + str(i)
    print('Reading ' + folder_name)
    folder_name = os.path.join(data_folder, folder_name)
    for j in range(1, 3):
        file_name = os.path.join(folder_name, 'S' + str(i) + '_E' + str(j) +'_A1.mat')
        mat_file = scipy.io.loadmat(file_name)
        #print(mat_file['emg'].shape, mat_file['restimulus'].shape)
        sti = mat_file['stimulus'][:, 0]
        action_step = np.where(np.abs(np.diff(sti)) > 0)[0]
        print(len(action_step))
        for ii in range(0, len(action_step) - 1, 2):
            data_all[i]['emg'].append(mat_file['emg'][action_step[ii] + 1 : action_step[ii + 1] + 1])
            #print(np.unique(mat_file['stimulus'][action_step[ii] + 1 : action_step[ii + 1] + 1][:, 0]))
            data_all[i]['label'].append(mat_file['stimulus'][action_step[ii] + 1 : action_step[ii + 1] + 1][1, 0])
            data_all[i]['ids'].append(i)
          
with open(os.path.join('/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EMG_Ninapro', 'emg_20.npy'), 'wb') as f:
    np.save(f, np.array(data_all, dtype=object))          

