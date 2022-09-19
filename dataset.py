
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
from utils import load_folds_data
import scipy
import scipy.io
import scipy.signal
from scipy.signal import butter, lfilter
import os
import random
from collections import Counter

class Chapman_Dataset(Dataset):
    def __init__(self, dataset_path = '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/ECG_chapman/contrastive_msml/leads/', 
        split_type='train', training_type = 'unsupervised', labeled_data = 0.25):
        super(Chapman_Dataset, self).__init__()
        # split_type: train, val, test 
        self.split_type = split_type
        self.training_type = training_type
        with open(dataset_path + 'frames_phases_chapman.pkl', 'rb') as f:
            frames = pickle.load(f)
        with open(dataset_path + 'labels_phases_chapman.pkl', 'rb') as f:
            labels = pickle.load(f)
        with open(dataset_path + 'pid_phases_chapman.pkl', 'rb') as f:
            pids = pickle.load(f)
        if (split_type == 'train') or (split_type == 'val') or (split_type == 'test'):
            self.frames = frames['ecg'][1][split_type]['All Terms'] 
            self.labels = labels['ecg'][1][split_type]['All Terms'] 
            self.pids = pids['ecg'][1][split_type]['All Terms'] 
        else:
            raise NotImplementedError('no'+ split_type) 
        if (split_type == 'train') and self.training_type == 'supervised':
            l = int(self.labels.shape[0] * labeled_data)
            self.frames = self.frames[:l]
            self.labels = self.labels[:l]
            self.pids = self.pids[:l]
        print(self.frames.shape)
        
    def __len__(self):
        return self.labels.shape[0]
  
    def __getitem__(self, index):
        frames = torch.tensor(self.frames[index]).float()
        mean_ecg = frames.mean(dim = 0, keepdim=True)
        d_ecg = frames.std(dim = 0, keepdim=True)
        frames = (frames - mean_ecg)/(d_ecg + 0.00001)
        if self.training_type == 'unsupervised':
            X = frames.reshape(1, frames.shape[0], frames.shape[1])
            Y = frames.reshape(1, frames.shape[0], frames.shape[1])
            return X, Y
        elif self.training_type == 'supervised': 
            X = frames.reshape(1, frames.shape[0], frames.shape[1])
            Y = torch.tensor(self.labels[index]).long()
            return  X, Y 
        else:
            raise NotImplementedError('no'+ self.training_type)

class EDF78_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset_path = '/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_EDF78/EEG_Sleep_Processed', 
        split_type='train', training_type = 'unsupervised',
        kfold = 5, kfold_id = 0, val_ratio = 0.2, labeled_data = 0.25):
        super(EDF78_Dataset, self).__init__()

        self.split_type = split_type
        self.training_type = training_type
        # load files
        self.split_type = split_type
        np_dataset = load_folds_data(np_data_path = dataset_path, n_folds = kfold)
        if split_type == 'train' or split_type == 'val':
            np_dataset = np_dataset[kfold_id][0]
        elif split_type == 'test':
            np_dataset = np_dataset[kfold_id][1]
        else:
            raise NotImplementedError('no'+ split_type) 

        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]
        s_id = np.ones(y_train.shape, dtype=int)*int(np_dataset[0][-10:-7])
        X_train = (X_train - X_train.mean())/(X_train.std() + 0.000001)

        for np_file in np_dataset[1:]:
            X_temp = np.load(np_file)["x"]
            X_temp = (X_temp - X_temp.mean())/(X_temp.std() + 0.000001)
            X_train = np.vstack((X_train, X_temp))
            y_temp = np.load(np_file)["y"]
            y_train = np.append(y_train, y_temp)
            s_id = np.append(s_id, np.ones(y_temp.shape, dtype=int)*int(np_file[-10:-7]))
            
        self.data_len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
        self.s_id = torch.from_numpy(s_id).long()

        if split_type == 'train':
            self.x_data = self.x_data[ : int(self.data_len * (1 - val_ratio)) if self.training_type == 'unsupervised' else int(self.data_len * (1 - val_ratio) * labeled_data)]
            self.y_data = self.y_data[ : int(self.data_len * (1 - val_ratio)) if self.training_type == 'unsupervised' else int(self.data_len * (1 - val_ratio) * labeled_data)]
            self.s_id = self.s_id[ : int(self.data_len * (1 - val_ratio))]
        if split_type == 'val':
            self.x_data = self.x_data[int(self.data_len * (1 - val_ratio)):]
            self.y_data = self.y_data[int(self.data_len * (1 - val_ratio)):]
            self.s_id = self.s_id[int(self.data_len * (1 - val_ratio)):]

        self.data_len = self.x_data.shape[0]
        print(self.x_data.shape, self.s_id.shape, self.y_data.shape)

    def __getitem__(self, index):
        frames = self.x_data[index]
        if self.training_type == 'unsupervised':
            return frames.unsqueeze(0), self.s_id[index]
        elif self.training_type == 'supervised': 
            return frames.unsqueeze(0), self.y_data[index]
        else:
            raise NotImplementedError('no'+ self.training_type)

    def __len__(self):
        return self.data_len

class EEG109_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, args, split_type):
        super(EEG109_Dataset, self).__init__()
        self.split_type = split_type
        self.training_type = args.training_type
        # load files
        self.split_type = split_type
        data = np.load(args.dataset_path, allow_pickle = True).item()
        """
        T0 corresponds to rest
        T1 corresponds to onset of motion (real or imagined) of
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
        T2 corresponds to onset of motion (real or imagined) of
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
        """
        random.seed(args.seed)
        subject_names = list([i for i in data.keys() if i not in [43, 88, 89, 92, 100, 104]])
        random.shuffle(subject_names)
        l = len(subject_names)
        self.eeg_signal = []
        self.labels = []
        self.ids = []
        if split_type == 'train':
            subjects = subject_names[:int(l * args.train_ratio)]
        elif split_type == 'val':
            #subjects = subject_names[:int(l * args.train_ratio)]
            subjects = subject_names[int(l * (1 - args.test_ratio - args.val_ratio)): int(l * (1 - args.test_ratio))]
        else:
            subjects = subject_names[int(l * (1 - args.test_ratio)): ]

        assert args.train_ratio + args.val_ratio + args.test_ratio <= 1
        print(subjects)

        for s in subjects:
          subject_signal = []
          for l in range(len(data[s])):
              data_temp = data[s][l][0][:646]
              if data[s][l][1] < 1:
                  continue
              if data_temp.shape[0] < 646:
                  subject_signal.append(torch.from_numpy(np.pad(data_temp, ((0, 646 - data_temp.shape[0]), (0, 0)), 'constant', constant_values=0)))
                  self.labels.append(data[s][l][1])
                  self.ids.append(s)
              else:
                  subject_signal.append(torch.from_numpy(data_temp))
                  self.labels.append(data[s][l][1])
                  self.ids.append(s)
          subject_signal = torch.stack(subject_signal)
          mean = subject_signal.mean(dim = [1, 2], keepdim = True)
          std = subject_signal.std(dim = [1, 2], keepdim = True)
          subject_signal = (subject_signal - mean)/std
          self.eeg_signal.append(subject_signal)

        print('Label Counter: ', dict(Counter(self.labels)))
        self.eeg_signal = torch.cat(self.eeg_signal).float()
        print(self.eeg_signal.shape)
        self.labels = torch.tensor(self.labels).long()
        self.ids = torch.tensor(self.ids).long()

        """if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                l_train = int(indexes_l * (1 - args.val_ratio))
                self.labels = self.labels[indexes[:l_train]]
                self.eeg_signal = self.eeg_signal[indexes[:l_train]]
                self.ids = self.ids[indexes[:l_train]]
                l = int(len(self.labels) * args.labeled_data)
                self.labels = self.labels[ : l]
                self.eeg_signal = self.eeg_signal[ : l]
                self.ids = self.ids[ : l]
            elif split_type == 'val':
                l_val = int(indexes_l * (1 - args.val_ratio))
                self.labels = self.labels[indexes[l_val : ]]
                self.eeg_signal = self.eeg_signal[indexes[l_val : ]]
                self.ids = self.ids[indexes[l_val : ]]"""

        if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)

            l = int(indexes_l * args.labeled_data)
            if split_type == 'train':
                self.labels = self.labels[indexes[:l]]
                self.eeg_signal = self.eeg_signal[indexes[:l]]
                self.ids = self.ids[indexes[:l]]
            else:
                self.labels = self.labels[indexes]
                self.eeg_signal = self.eeg_signal[indexes]
                self.ids = self.ids[indexes]
        
        print(self.eeg_signal.shape, self.labels.shape)
        
    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        if self.training_type == 'unsupervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = [self.labels[index] - 1, self.ids[index]]#[self.labels[index], self.ids[index]]
        elif self.training_type == 'supervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = self.labels[index] - 1
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y 

class EMGDB7_Dataset(Dataset):
    def __init__(self, args, split_type):
        super(EMGDB7_Dataset, self).__init__()
        self.split_type = split_type
        self.training_type = args.training_type
        self.split_type = split_type
        data = np.load(args.dataset_path, allow_pickle = True).item()
        random.seed(args.seed)
        subject_names = list(data.keys())
        random.shuffle(subject_names)
        l = len(subject_names)
        self.emg_signal = []
        self.labels = []
        self.ids = []
        if split_type == 'train':
            subjects = subject_names[:int(l * (args.train_ratio))]
        elif split_type == 'val':
            subjects = subject_names[:int(l * (args.train_ratio))]
        else:
            subjects = subject_names[int(l * (1 - args.test_ratio)):]
        print(subjects)

        b, a = butter(N = 4, Wn = 20, btype='highpass', analog=False, fs=2000, output = 'sos')#
        for s in subjects:
          for l in range(len(data[s]['label'])):
              data_temp = data[s]
              emgs = data_temp['emg'][l]
              seq_length = emgs.shape[0]
              assert seq_length >= args.seq_length
              split_step = args.split_stride if self.training_type == 'supervised' else args.seq_length
              for i in range(args.split_stride * 2, seq_length - args.seq_length - args.split_stride, split_step): 
                  emg_single = lfilter(b, a, emgs[i: i + args.seq_length])
                  self.emg_signal.append(torch.from_numpy(emg_single))
                  self.labels.append(data_temp['label'][l])
                  self.ids.append(s)
         
        self.emg_signal = torch.stack(self.emg_signal).float()
        self.labels = torch.tensor(self.labels).long()
        self.ids = torch.tensor(self.ids).long()

        """To be deleted"""
        """"g = torch.Generator()
        g.manual_seed(0)
        indexes = torch.randperm(self.labels.shape[0], generator = g)
        l = self.labels.shape[0]
        if split_type == 'train':
            self.labels = self.labels[indexes[:int(l * (1 - args.test_ratio - args.val_ratio))]]
            self.emg_signal = self.emg_signal[indexes[:int(l * (1 - args.test_ratio - args.val_ratio))]]
        elif split_type == 'val':
            self.labels = self.labels[indexes[int(l * (1 - args.test_ratio - args.val_ratio)): int(l * (1 - args.test_ratio))]]
            self.emg_signal = self.emg_signal[indexes[int(l * (1 - args.test_ratio - args.val_ratio)): int(l * (1 - args.test_ratio))]]
        else:
            self.labels = self.labels[indexes[int(l * (1 - args.test_ratio)): ]]
            self.emg_signal = self.emg_signal[indexes[int(l * (1 - args.test_ratio)): ]]"""
        """To be deleted"""

        if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                l_train = int(indexes_l * (1 - args.val_ratio))
                self.labels = self.labels[indexes[:l_train]]
                self.emg_signal = self.emg_signal[indexes[:l_train]]
                self.ids = self.ids[indexes[:l_train]]
                if self.training_type =='supervised':
                    l = int(len(self.labels) * args.labeled_data)
                    self.labels = self.labels[ : l]
                    self.emg_signal = self.emg_signal[ : l]
                    self.ids = self.ids[ :l]
            elif split_type == 'val':
                l_val = int(indexes_l * (1 - args.val_ratio))
                self.labels = self.labels[indexes[l_val : ]]
                self.emg_signal = self.emg_signal[indexes[l_val : ]]
                self.ids = self.ids[indexes[l_val : ]]
            
        mean = self.emg_signal.mean(dim = [1, 2], keepdim = True)
        std = self.emg_signal.std(dim = [1, 2], keepdim = True)
        self.emg_signal = (self.emg_signal - mean)/std
        """max_emg = self.emg_signal.amax(dim = (1, 2), keepdim = True)
        min_emg = self.emg_signal.amin(dim = (1, 2), keepdim = True)
        self.emg_signal = (self.emg_signal - min_emg)/(max_emg - min_emg)"""
        print(self.emg_signal.shape, self.ids.shape, self.labels.shape)
        
    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        if self.training_type == 'unsupervised':
            X = self.emg_signal[index].unsqueeze(0)
            Y = self.ids[index]
        elif self.training_type == 'supervised':
            X = self.emg_signal[index].unsqueeze(0)
            Y = self.labels[index] - 1
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y 

class EEG2a_Dataset(Dataset):
    def __init__(self, split_type, args):
        super(EEG2a_Dataset, self).__init__()
        random.seed(args.seed)
        all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        eeg_signal = []
        labels = []
        ids = []
        data_class = ['data1', 'data2', 'data3', 'data4']
        self.training_type = args.training_type
        if split_type == 'train' or split_type == 'val':
            split_id = all_ids[:int(len(all_ids) * (args.train_ratio))]
        elif split_type == 'test':
            split_id = all_ids[int(len(all_ids) * (1 - args.test_ratio)):]
        file_num_range = ['A0' + str(i) + 'T.mat'for i in split_id]
        file_num_range.extend(['A0' + str(i) + 'E.mat'for i in split_id])
        print(file_num_range)
        for set_num in file_num_range:
            file_name = set_num
            subject_id = int(file_name[2])
            mat = scipy.io.loadmat(os.path.join(args.dataset_path, file_name))
            for i in range(4): # each file has 4 classes
                data_num = mat[data_class[i]].shape[2]
                for j in range(data_num): # each class has a specific number of data
                    eeg_signal.append(mat[data_class[i]][:,:,j])
                    labels.append(i)
                    ids.append(subject_id)
        self.labels = torch.tensor(np.array(labels)).long()
        self.eeg_signal = torch.tensor(np.array(eeg_signal)).float()
        self.ids = torch.tensor(np.array(ids)).long()
        if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                l_train = int(indexes_l * (1 - args.val_ratio))
                self.labels = self.labels[indexes[:l_train]]
                self.eeg_signal = self.eeg_signal[indexes[:l_train]]
                self.ids = self.ids[indexes[:l_train]]
                if self.training_type =='supervised':
                    l = int(len(self.labels) * args.labeled_data)
                    self.labels = self.labels[ : l]
                    self.eeg_signal = self.eeg_signal[ : l]
                    self.ids = self.ids[ : l]
            elif split_type == 'val':
                l_val = int(indexes_l * (1 - args.val_ratio))
                self.labels = self.labels[indexes[l_val : ]]
                self.eeg_signal = self.eeg_signal[indexes[l_val : ]]
                self.ids = self.ids[indexes[l_val : ]]

        mean = self.eeg_signal.mean(dim = [1, 2], keepdim = True)
        std = self.eeg_signal.std(dim = [1, 2], keepdim = True)
        self.eeg_signal = (self.eeg_signal - mean)/std
        print(self.eeg_signal.shape)

    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        if self.training_type == 'unsupervised':
            X = self.eeg_signal[index].unsqueeze(0)
            """To be deleted"""
            Y = [self.labels[index],self.ids[index]]
        elif self.training_type == 'supervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = self.labels[index] - 1
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y 


    
"""Test"""
import matplotlib.pyplot as plt
if __name__ == "__main__":
    d = EMGDB7_Dataset(split_type = 'train')
    plt.figure(figsize = (40, 10))
    plt.plot(d.__getitem__(0)[0][0, :, 0].numpy())
    plt.savefig('test.jpg')
    d.__getitem__(10)
    d.__getitem__(1000)

