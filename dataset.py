
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
"""import moabb
from moabb import datasets
from moabb.paradigms import MotorImagery
from moabb.paradigms import LeftRightImagery
from moabb.paradigms import FilterBankMotorImagery"""
import mne

class EEG109_Dataset(Dataset):
    """
        T0 corresponds to rest
        T1 corresponds to onset of motion (real or imagined) of
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
        T2 corresponds to onset of motion (real or imagined) of
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
    """
    def __init__(self, args, split_type):
        super(EEG109_Dataset, self).__init__()
        assert args.train_ratio_all >= args.train_ratio
        assert args.train_ratio_all + args.val_ratio + args.test_ratio <= 1
        assert args.train_ratio_all >= args.train_ratio
        assert args.labeled_data_all >= args.labeled_data
        assert args.train_ratio_all == args.train_ratio or args.labeled_data_all == args.labeled_data
        self.split_type = split_type
        self.training_type = args.training_type
        # load files
        self.split_type = split_type
        data = np.load(args.dataset_path, allow_pickle = True).item()
        random.seed(args.seed)
        subject_names = list([i for i in data.keys() if i not in [43, 88, 89, 92, 100, 104]])
        random.shuffle(subject_names)
        l = len(subject_names)
        self.eeg_signal = []
        self.labels = []
        self.ids = []
        self.test_labels = []
        label_flag = []
        if split_type == 'train':
            subjects = subject_names[:int(l * args.train_ratio_all)]
            label_flag = [1] * int(l * args.train_ratio)
            # args.train_ratio_all - args.train_ratio subjects will be used but there is no labels
            zero_flag = [0] * (int(l * args.train_ratio_all) - int(l * args.train_ratio))
            label_flag.extend(zero_flag)
        elif split_type == 'val':
            #subjects = subject_names[:int(l * args.train_ratio)]
            subjects = subject_names[int(l * (1 - args.test_ratio - args.val_ratio)): int(l * (1 - args.test_ratio))]
            label_flag = [1] * (int(l * (1 - args.test_ratio)) - int(l * (1 - args.test_ratio - args.val_ratio)))
        else:
            subjects = subject_names[int(l * (1 - args.test_ratio)): ]
            label_flag = [1] * (l - int(l * (1 - args.test_ratio)))

        print(subjects)
        print(label_flag)
        for i in range(len(subjects)):
            s = subjects[i]
            subject_signal = []
            for l in range(len(data[s])):
                data_temp = data[s][l][0][:646]
                if data[s][l][1] < 1:
                    continue
                if data_temp.shape[0] < 646:
                    subject_signal.append(torch.from_numpy(np.pad(data_temp, ((0, 646 - data_temp.shape[0]), (0, 0)), 'constant', constant_values=0)))
                else:
                    subject_signal.append(torch.from_numpy(data_temp))
                if label_flag[i] == 0:
                    self.labels.append(-1)
                    self.test_labels.append(data[s][l][1])
                else:
                    self.labels.append(data[s][l][1])
                    self.test_labels.append(data[s][l][1])
                self.ids.append(s)
            subject_signal = torch.stack(subject_signal)
            mean = subject_signal.mean(dim = [0, 1], keepdim = True)
            std = subject_signal.std(dim = [0, 1], keepdim = True)
            subject_signal = (subject_signal - mean)/std
            self.eeg_signal.append(subject_signal)

        print('Label Counter: ', dict(Counter(self.labels)))
        self.eeg_signal = torch.cat(self.eeg_signal).float()
        print(self.eeg_signal.shape)
        self.labels = torch.tensor(self.labels).long()
        self.ids = torch.tensor(self.ids).long()
        self.test_labels = torch.tensor(self.test_labels).long()

        if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                l_all = int(indexes_l * args.labeled_data_all)
                self.labels = self.labels[indexes[:l_all]]
                self.eeg_signal = self.eeg_signal[indexes[:l_all]]
                self.ids = self.ids[indexes[:l_all]]
                self.test_labels = self.test_labels[indexes[:l_all]]
                #l_all -l data will be used but without an label
                l = int(indexes_l * args.labeled_data)
                self.labels[l:l_all] = -1
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
            Y = [self.labels[index] - 1, self.ids[index], self.true_labels[index] - 1]#[self.labels[index], self.ids[index]]
        elif self.training_type == 'supervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = self.labels[index] - 1
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y 

class Cho2017_Dataset(Dataset):
    def __init__(self, args, split_type):
        super(Cho2017_Dataset, self).__init__()
        assert args.train_ratio_all >= args.train_ratio
        assert args.train_ratio_all + args.val_ratio + args.test_ratio <= 1
        assert args.train_ratio_all >= args.train_ratio
        assert args.labeled_data_all >= args.labeled_data
        assert args.train_ratio_all == args.train_ratio or args.labeled_data_all == args.labeled_data
        self.split_type = split_type
        self.training_type = args.training_type
        self.label_map = {'left_hand': 0, 'right_hand': 1}
        # load files
        self.split_type = split_type
        self.paradigm = LeftRightImagery()
        data = datasets.Cho2017()
        random.seed(args.seed)
        subject_names = list([i for i in range(1, 53) if i not in [32, 20, 33, 46, 49]])
        random.shuffle(subject_names)
        l = len(subject_names)
        mne.set_config('MNE_DATASETS_CHO2017_PATH', args.dataset_path)
        mne.set_config('MNE_DATASETS_GIGADB_PATH', args.dataset_path)

        self.eeg_signal = []
        self.labels = []
        self.ids = []
        
        label_flag = []
        if split_type == 'train':
            subjects = subject_names[:int(l * args.train_ratio_all)]
            label_flag = [1] * int(l * args.train_ratio)
            # args.train_ratio_all - args.train_ratio subjects will be used but there is no labels
            zero_flag = [0] * (int(l * args.train_ratio_all) - int(l * args.train_ratio))
            label_flag.extend(zero_flag)
        elif split_type == 'val':
            #subjects = subject_names[:int(l * args.train_ratio)]
            subjects = subject_names[int(l * (1 - args.test_ratio - args.val_ratio)): int(l * (1 - args.test_ratio))]
            label_flag = [1] * (int(l * (1 - args.test_ratio)) - int(l * (1 - args.test_ratio - args.val_ratio)))
        else:
            subjects = subject_names[int(l * (1 - args.test_ratio)): ]
            label_flag = [1] * (l - int(l * (1 - args.test_ratio)))

        print(subjects)
        print(label_flag)
        for i in range(len(subjects)):
            s = subjects[i]
            subject_signal = []
            X, labels_o, meta = self.paradigm.get_data(dataset=data, subjects=[s])
            for l in range(X.shape[0]):
                subject_signal.append(torch.from_numpy(X[l]))
                if label_flag[i] == 0:
                    self.labels.append(-1)
                else:
                    self.labels.append(self.label_map[labels_o[l]])
                self.ids.append(s) 
            subject_signal = torch.stack(subject_signal)
            subject_signal = subject_signal.transpose(1, 2)
            mean = subject_signal.mean(dim = [0, 1], keepdim = True)
            std = subject_signal.std(dim = [0, 1], keepdim = True)
            subject_signal = (subject_signal - mean)/std
            self.eeg_signal.append(subject_signal)

        print('Label Counter: ', dict(Counter(self.labels)))
        self.eeg_signal = torch.cat(self.eeg_signal).float()
        print(self.eeg_signal.shape)
        self.labels = torch.tensor(self.labels).long()
        self.ids = torch.tensor(self.ids).long()

        if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                l_all = int(indexes_l * args.labeled_data_all)
                self.labels = self.labels[indexes[:l_all]]
                self.eeg_signal = self.eeg_signal[indexes[:l_all]]
                self.ids = self.ids[indexes[:l_all]]
                #l_all -l data will be used but without an label
                l = int(indexes_l * args.labeled_data)
                self.labels[l:l_all] = -1
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
            Y = [self.labels[index], self.ids[index]]
        elif self.training_type == 'supervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = self.labels[index]
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y

class HaLT12_Dataset(Dataset):
    def __init__(self, args, split_type):
        super(HaLT12_Dataset, self).__init__()
        assert args.train_ratio_all >= args.train_ratio
        assert args.train_ratio_all + args.test_ratio <= 1
        assert args.train_ratio_all >= args.train_ratio
        assert args.labeled_data_all >= args.labeled_data
        assert args.train_ratio_all == args.train_ratio or args.labeled_data_all == args.labeled_data
        self.split_type = split_type
        self.training_type = args.training_type
        self.label_map = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4 ,
                          0: -99, 3: -99, 91: -99, 92: -99, 99: -99}
        # load files
        self.split_type = split_type
        data = np.load(args.dataset_path, allow_pickle = True).item()
        random.seed(args.seed)
        subject_names = list(range(1, 13))
        random.shuffle(subject_names)
        l = len(subject_names)
      
        self.eeg_signal = []
        self.labels = []
        self.ids = []
        
        label_flag = []
        if split_type == 'train':
            subjects = subject_names[:int(l * args.train_ratio_all)]
            label_flag = [1] * int(l * args.train_ratio)
            # args.train_ratio_all - args.train_ratio subjects will be used but there is no labels
            zero_flag = [0] * (int(l * args.train_ratio_all) - int(l * args.train_ratio))
            label_flag.extend(zero_flag)
        elif split_type == 'val':
            subjects = subject_names[:int(l * args.train_ratio_all)]
            label_flag = [1] * int(l * args.train_ratio)
            # args.train_ratio_all - args.train_ratio subjects will be used but there is no labels
            zero_flag = [0] * (int(l * args.train_ratio_all) - int(l * args.train_ratio))
            label_flag.extend(zero_flag)
        else:
            subjects = subject_names[int(l * (1 - args.test_ratio)): ]
            label_flag = [1] * (l - int(l * (1 - args.test_ratio)))

        print(subjects)
        print(label_flag)
        for i in range(len(subjects)):
            s = subjects[i]
            subject_signal = []
            labels = []
            ids = []
            for l in range(len(data[s])):
                label = data[s][l][1][0]
                if self.label_map[label] >= 0:
                    subject_signal.append(data[s][l][0][:200])
                    if label_flag[i] == 0:
                        labels.append(-1)
                    else:
                        labels.append(self.label_map[label])
                    ids.append(s) 
            
            subject_signal = torch.stack(subject_signal)
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(subject_signal.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                indexes = indexes[:int(indexes_l * (1 - args.val_ratio))]
            if split_type == 'val':
                indexes = indexes[int(indexes_l * (1 - args.val_ratio)):]
            
            subject_signal = subject_signal[indexes]
            labels = torch.tensor(labels)[indexes].tolist()
            ids = torch.tensor(ids)[indexes].tolist()
            
            self.labels.extend(labels)
            self.ids.extend(ids)
            
            mean = subject_signal.mean(dim = [0, 1], keepdim = True)
            std = subject_signal.std(dim = [0, 1], keepdim = True)
            subject_signal = (subject_signal - mean)/std
            self.eeg_signal.append(subject_signal)

        print('Label Counter: ', dict(Counter(self.labels)))
        self.eeg_signal = torch.cat(self.eeg_signal).float()
        print(self.eeg_signal.shape)
        self.labels = torch.tensor(self.labels).long()
        self.ids = torch.tensor(self.ids).long()

        if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                l_all = int(indexes_l * args.labeled_data_all)
                self.labels = self.labels[indexes[:l_all]]
                self.eeg_signal = self.eeg_signal[indexes[:l_all]]
                self.ids = self.ids[indexes[:l_all]]
                #l_all -l data will be used but without an label
                l = int(indexes_l * args.labeled_data)
                self.labels[l:l_all] = -1
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
            Y = [self.labels[index], self.ids[index]]
        elif self.training_type == 'supervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = self.labels[index]
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y

class Shin2017_Dataset(Dataset):
    def __init__(self, args, split_type):
        super(Shin2017_Dataset, self).__init__()
        assert args.train_ratio_all >= args.train_ratio
        assert args.train_ratio_all + args.val_ratio + args.test_ratio <= 1
        assert args.train_ratio_all >= args.train_ratio
        assert args.labeled_data_all >= args.labeled_data
        assert args.train_ratio_all == args.train_ratio or args.labeled_data_all == args.labeled_data
        self.split_type = split_type
        self.training_type = args.training_type
        self.label_map = {'left_hand': 0, 
                          'right_hand': 1 
                        }
        # load files
        self.split_type = split_type
        self.paradigm = MotorImagery()
        data = datasets.Shin2017A(accept=True)
        random.seed(args.seed)
        subject_names = list([i for i in range(1, 30)])
        random.shuffle(subject_names)
        l = len(subject_names)
        mne.set_config('MNE_DATASETS_BBCIFNIRS_PATH', args.dataset_path)

        self.eeg_signal = []
        self.labels = []
        self.ids = []
        
        label_flag = []
        if split_type == 'train':
            subjects = subject_names[:int(l * args.train_ratio_all)]
            label_flag = [1] * int(l * args.train_ratio)
            # args.train_ratio_all - args.train_ratio subjects will be used but there is no labels
            zero_flag = [0] * (int(l * args.train_ratio_all) - int(l * args.train_ratio))
            label_flag.extend(zero_flag)
        elif split_type == 'val':
            #subjects = subject_names[:int(l * args.train_ratio)]
            subjects = subject_names[int(l * (1 - args.test_ratio - args.val_ratio)): int(l * (1 - args.test_ratio))]
            label_flag = [1] * (int(l * (1 - args.test_ratio)) - int(l * (1 - args.test_ratio - args.val_ratio)))
        else:
            subjects = subject_names[int(l * (1 - args.test_ratio)): ]
            label_flag = [1] * (l - int(l * (1 - args.test_ratio)))

        print(subjects)
        print(label_flag)
        for i in range(len(subjects)):
            s = subjects[i]
            subject_signal = []
            X, labels_o, meta = self.paradigm.get_data(dataset=data, subjects=[s])
            for l in range(X.shape[0]):
                single_trail = X[l]
                for j in range(0, 2000, args.seq_length):
                    subject_signal.append(torch.from_numpy(single_trail[:, j : j + args.seq_length]))
                    if label_flag[i] == 0:
                        self.labels.append(-1)
                    else:
                        self.labels.append(self.label_map[labels_o[l]])
                    self.ids.append(s) 
            subject_signal = torch.stack(subject_signal)
            subject_signal = subject_signal.transpose(1, 2)
            mean = subject_signal.mean(dim = [0, 1], keepdim = True)
            std = subject_signal.std(dim = [0, 1], keepdim = True)
            subject_signal = (subject_signal - mean)/std
            self.eeg_signal.append(subject_signal)

        print('Label Counter: ', dict(Counter(self.labels)))
        self.eeg_signal = torch.cat(self.eeg_signal).float()
        print(self.eeg_signal.shape)
        self.labels = torch.tensor(self.labels).long()
        self.ids = torch.tensor(self.ids).long()

        if split_type == 'train' or split_type == 'val':
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(self.labels.shape[0], generator = g)
            indexes_l = len(indexes)
            if split_type == 'train':
                l_all = int(indexes_l * args.labeled_data_all)
                self.labels = self.labels[indexes[:l_all]]
                self.eeg_signal = self.eeg_signal[indexes[:l_all]]
                self.ids = self.ids[indexes[:l_all]]
                #l_all -l data will be used but without an label
                l = int(indexes_l * args.labeled_data)
                self.labels[l:l_all] = -1
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
            Y = [self.labels[index], self.ids[index]]
        elif self.training_type == 'supervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = self.labels[index]
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y
        
class EEG2a_Dataset(Dataset):
    def __init__(self, split_type, args):
        super(EEG2a_Dataset, self).__init__()
        random.seed(args.seed)
        self.split_type = split_type
        self.training_type = args.training_type
        # load files
        self.split_type = split_type
        random.seed(args.seed)
        subject_names = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.shuffle(subject_names)
        l = len(subject_names)
        self.eeg_signal = []
        self.labels = []
        self.ids = []
        self.test_labels = []
        self.reorder_index = []
        label_flag = []

        self.data_class = ['data1', 'data2', 'data3', 'data4']
        if split_type == 'train':
            subjects = subject_names[:int(l * args.train_ratio_all)]
            label_flag = [1] * int(l * args.train_ratio)
            # args.train_ratio_all - args.train_ratio subjects will be used but there is no labels
            zero_flag = [0] * (int(l * args.train_ratio_all) - int(l * args.train_ratio))
            label_flag.extend(zero_flag)
        elif split_type == 'val':
            subjects = subject_names[int(l * (1 - args.test_ratio - args.val_ratio)): int(l * (1 - args.test_ratio))]
            label_flag = [1] * (int(l * (1 - args.test_ratio)) - int(l * (1 - args.test_ratio - args.val_ratio)))
        else:
            subjects = subject_names[int(l * (1 - args.test_ratio)): ]
            label_flag = [1] * (l - int(l * (1 - args.test_ratio)))
        
        print(subjects)
        print(label_flag)
        for i in range(len(subjects)):
            s = subjects[i]
            subject_signal = []
            labels = []
            matT = scipy.io.loadmat(os.path.join(args.dataset_path, 'A0' + str(s) + 'T.mat'))
            matE = scipy.io.loadmat(os.path.join(args.dataset_path, 'A0' + str(s) + 'E.mat'))
            for j in range(4): # each file has 4 classes
                subject_signal.append(torch.tensor(matT[self.data_class[j]]).permute(2, 0, 1))
                subject_signal.append(torch.tensor(matE[self.data_class[j]]).permute(2, 0, 1))
                total_l = (matT[self.data_class[j]].shape[2] + matE[self.data_class[j]].shape[2])
                labels.extend([j] * total_l)

            subject_signal = torch.cat(subject_signal)
            mean = subject_signal.mean(dim = [0, 1], keepdim = True)
            std = subject_signal.std(dim = [0, 1], keepdim = True)
            subject_signal = (subject_signal - mean)/std
            labels = torch.tensor(labels)
            print(subject_signal.shape, labels.shape)
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(subject_signal.shape[0], generator = g)
            indexes_l = len(indexes)
            if self.split_type == 'train':
                l_all = int(indexes_l * args.labeled_data_all)
                labels = labels[indexes[:l_all]]
                subject_signal = subject_signal[indexes[:l_all]]
                #l_all -l data will be used but without an label
                l = int(indexes_l * args.labeled_data)
                labels[l:l_all] = -1
            self.eeg_signal.append(subject_signal)
            self.labels.append(labels)

        self.eeg_signal = torch.cat(self.eeg_signal).float()
        self.labels = torch.cat(self.labels).long()
        print(self.eeg_signal.shape, self.labels.shape)
        print('Label Counter: ', dict(Counter(self.labels.numpy())))
        g = torch.Generator()
        g.manual_seed(args.seed)
        indexes = torch.randperm(self.labels.shape[0], generator = g)
        self.labels = self.labels[indexes]
        self.eeg_signal = self.eeg_signal[indexes]
        

    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        X = self.eeg_signal[index].unsqueeze(0)
        Y = self.labels[index]
        return X, Y
