import mne
from mne import Epochs, pick_types, events_from_annotations
import os
import numpy as np
import pandas as pd

"""
T0 corresponds to rest task id 0
T1 corresponds to onset of motion (real or imagined) of
    the left fist (in runs 3, 4, 7, 8, 11, and 12) task id: 1
    both fists (in runs 5, 6, 9, 10, 13, and 14) task id: 2
T2 corresponds to onset of motion (real or imagined) of
    the right fist (in runs 3, 4, 7, 8, 11, and 12) task id: 3
    both feet (in runs 5, 6, 9, 10, 13, and 14) task id: 4
"""
total_data = {}
task_dict = {}
run0 = [1, 2]
run1 = [4, 8, 12]#[3, 4, 7, 8, 11, 12][3, 7, 11]#
run2 = [6, 10, 14]#[5, 6, 9, 10, 13, 14][5, 9, 13]#
total_run = []
total_run.extend(run1)
total_run.extend(run2)
for r in run0:
  task_dict['r' + str(r) + '_t1'] = 0
for r in run1:
  task_dict['r' + str(r)+'_t1'] = 0
  task_dict['r' + str(r)+'_t2'] = 1
  task_dict['r' + str(r)+'_t3'] = 3
for r in run2:
  task_dict['r' + str(r)+'_t1'] = 0
  task_dict['r' + str(r)+'_t2'] = 2
  task_dict['r' + str(r)+'_t3'] = 4

print(task_dict)

data_folder = "/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/files"
for s in range(1, 110):# 110
    total_data[s] = []
    for r in total_run:# 15
        file_name = 'S' + str("{0:03d}".format(s)) + '/S' + str("{0:03d}".format(s)) + 'R' + str("{0:02d}".format(r)) + '.edf'
        file = os.path.join(data_folder, file_name)
        data = mne.io.read_raw_edf(file)
        d = data.to_data_frame()
        channels = data.ch_names
        events, names = mne.events_from_annotations(data)
        l = events.shape[0]
        events = np.append(events, [[len(d) + 1, 0, -1]], axis=0)
        for i in range(l):
            eeg_try = d.loc[events[i][0]:events[i + 1][0], channels].to_numpy()
            total_data[s].append([eeg_try, task_dict['r' + str(r) + '_t' + str(events[i][2])]])

with open(os.path.join('/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1', 'eeg_109_imagery.npy'), 'wb') as f:
    np.save(f, np.array(total_data, dtype=object))      