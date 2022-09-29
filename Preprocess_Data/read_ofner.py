import moabb
from moabb import datasets
from moabb.paradigms import MotorImagery
from moabb.paradigms import LeftRightImagery
from moabb.paradigms import FilterBankMotorImagery
import mne 
import os
paradigm = MotorImagery(n_classes = 6)
data = datasets.Ofner2017()
mne.set_config('MNE_DATASETS_OFNER2017_PATH', '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/Ofner2017')
mne.set_config('MNE_DATASETS_GIGADB_PATH', '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/Ofner2017')
mne.set_config('MNE_DATASETS_UPPERLIMB_PATH', '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/Ofner2017')
total_data = {}
for i in range(1, 16):
    X, labels_o, _ = paradigm.get_data(dataset=data, subjects=[i])
    total_data[i] = [X, labels_o]

import numpy as np
with open(os.path.join('/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/Ofner2017', 'Ofner2017.npy'), 'wb') as f:
    np.save(f, np.array(total_data, dtype=object))  