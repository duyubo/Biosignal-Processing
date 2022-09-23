
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

class Base_Dataset(Dataset):
    def __init__(self, args, split_type):
        super(Base_Dataset, self).__init__()
        assert args.train_ratio_all >= args.train_ratio
        assert args.train_ratio_all + args.val_ratio + args.test_ratio <= 1
        assert args.train_ratio_all >= args.train_ratio
        assert args.labeled_data_all >= args.labeled_data
        assert args.train_ratio_all == args.train_ratio or args.labeled_data_all == args.labeled_data
        self.split_type = split_type
        self.training_type = args.training_type
        # load files
        self.split_type = split_type
        x_c, y_c = torch.meshgrid(torch.arange(args.final_dim), torch.arange(args.final_dim))
        self.map_indexes = torch.stack([x_c.triu(diagonal = 1), y_c.triu(diagonal = 1)]).permute(1, 2, 0)

    def reorganize_data(self, subjects, label_flag, args, data = None):
        length_flag = 0
        for i in range(len(subjects)):
            s = subjects[i]
            subject_signal, labels, test_labels, ids = self.load_data(label_flag = label_flag, args = args, s = s, i = i, data = data)
            mean = subject_signal.mean(dim = [0, 1], keepdim = True)
            std = subject_signal.std(dim = [0, 1], keepdim = True)
            subject_signal = (subject_signal - mean)/std
            labels = torch.tensor(labels)
            ids = torch.tensor(ids)
            test_labels = torch.tensor(test_labels)
            #print(subject_signal.shape, labels.shape, ids.shape, test_labels.shape)
            g = torch.Generator()
            g.manual_seed(args.seed)
            indexes = torch.randperm(subject_signal.shape[0], generator = g)
            indexes_l = len(indexes)
            if self.split_type == 'train':
                l_all = int(indexes_l * args.labeled_data_all)
                labels = labels[indexes[:l_all]]
                test_labels = test_labels[indexes[:l_all]]
                subject_signal = subject_signal[indexes[:l_all]]
                ids = ids[indexes[:l_all]]
                #l_all -l data will be used but without an label
                l = int(indexes_l * args.labeled_data)
                labels[l:l_all] = -1
            
            l_x, l_y = torch.meshgrid(labels, labels)
            i_x, i_y = torch.meshgrid(torch.arange(length_flag, length_flag + labels.shape[0]), torch.arange(length_flag, length_flag + labels.shape[0]))
            reorder_index = torch.stack((i_x, i_y)).permute(1, 2, 0)[i_x != i_y].reshape(-1, 2)
            self.reorder_index.append(reorder_index)
            labels = torch.stack((l_x, l_y)).permute(1, 2, 0)
            l_metrix_or = torch.logical_or(l_x == -1, l_y == -1)
            binary_metrix = (l_x == l_y) * (~l_metrix_or)
            labels[binary_metrix] = 0
            labels[l_metrix_or] = -1
            unique_pair, labels = torch.unique(torch.cat([torch.tensor([[-1, -1]]), self.map_indexes.reshape(-1, 2), labels[i_x != i_y].sort()[0]]), dim = 0, return_inverse = True)
            #print(unique_pair.shape)
            labels = labels[args.final_dim * args.final_dim + 1:]
            labels = labels - 1 
            self.eeg_signal.append(subject_signal)
            self.labels.append(labels)
            ids = [s] * labels.shape[0]
            self.ids.extend(ids)
            test_labels = torch.stack(torch.meshgrid(test_labels, test_labels)).permute(1, 2, 0)[i_x != i_y]
            self.test_labels.append(test_labels)
            length_flag += subject_signal.shape[0]
            """print("subject_signal.shape", subject_signal.shape, 
                "reorder_index.shape", reorder_index.shape, 
                "labels.shape", labels.shape, 
                "len(ids)", len(ids), 
                "test_labels.shape", test_labels.shape)"""
        
        self.eeg_signal = torch.cat(self.eeg_signal).float()
        print(self.eeg_signal.shape)
        self.reorder_index = torch.cat(self.reorder_index)
        self.labels = torch.cat(self.labels).long()
        self.ids = torch.tensor(self.ids).long()
        self.test_labels = torch.cat(self.test_labels).long()
        print('Label Counter: ', dict(Counter(self.labels.numpy())))

        g = torch.Generator()
        g.manual_seed(args.seed)
        indexes = torch.randperm(self.labels.shape[0], generator = g)
        self.reorder_index = self.reorder_index[indexes]
        self.labels = self.labels[indexes]
        self.ids = self.ids[indexes]
        self.test_labels = self.test_labels[indexes]
                   
        print(self.eeg_signal.shape, self.reorder_index.shape, self.ids.shape, self.labels.shape)

    def __len__(self):
        if self.split_type == "train":
            return int(len(self.labels) * 1)
        else:
            return int(len(self.labels))
  
    def __getitem__(self, index):
        if self.training_type == 'unsupervised':
            X = torch.stack((self.eeg_signal[self.reorder_index[index, 0]].unsqueeze(0), 
                            self.eeg_signal[self.reorder_index[index, 1]].unsqueeze(0)))
            Y = [self.labels[index], self.ids[index], self.test_labels[index]]
        elif self.training_type == 'supervised':
            X = self.eeg_signal[index].unsqueeze(0)
            Y = self.labels[index]
        else:
            raise NotImplementedError('no'+ self.training_type)
        return X, Y


class EEG109_Dataset(Base_Dataset):
    # Initialize your data, download, etc.
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
        super(EEG109_Dataset, self).__init__(args, split_type)
        
        data = np.load(args.dataset_path, allow_pickle = True).item()
        random.seed(args.seed)
        subject_names = list([i for i in data.keys() if i not in [43, 88, 89, 92, 100, 104]])
        random.shuffle(subject_names)
        l = len(subject_names)
        self.eeg_signal = []
        self.labels = []
        self.ids = []
        self.test_labels = []
        self.reorder_index = []
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

        self.reorganize_data(subjects = subjects, label_flag = label_flag, args = args, data = data)

    def load_data(self, label_flag, args, s, i, data):
        subject_signal = []
        labels = []
        test_labels = []
        ids = []
        for l in range(len(data[s])):
            data_temp = data[s][l][0][:646]
            if data[s][l][1] < 1:
                continue
            if data_temp.shape[0] < 646:
                subject_signal.append(torch.from_numpy(np.pad(data_temp, ((0, 646 - data_temp.shape[0]), (0, 0)), 'constant', constant_values=0)))
            else:
                subject_signal.append(torch.from_numpy(data_temp))
            if label_flag[i] == 0:
                labels.append(-1)
                test_labels.append(data[s][l][1] - 1)
            else:
                labels.append(data[s][l][1] - 1)
                test_labels.append(data[s][l][1] - 1)
            ids.append(s) 
        subject_signal = torch.stack(subject_signal)
        return subject_signal, labels, test_labels, ids

class EEG2a_Dataset(Base_Dataset):
    def __init__(self, split_type, args):
        super(EEG2a_Dataset, self).__init__(args, split_type)
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
        self.reorganize_data(subjects = subjects, label_flag = label_flag, args = args, data = None)

    def load_data(self, label_flag, args, s, i, data):
        subject_signal = []
        labels = []
        test_labels = []
        ids = []
        matT = scipy.io.loadmat(os.path.join(args.dataset_path, 'A0' + str(s) + 'T.mat'))
        matE = scipy.io.loadmat(os.path.join(args.dataset_path, 'A0' + str(s) + 'E.mat'))
        for j in range(4): # each file has 4 classes
            subject_signal.append(torch.tensor(matT[self.data_class[j]]).permute(2, 0, 1))
            subject_signal.append(torch.tensor(matE[self.data_class[j]]).permute(2, 0, 1))
            total_l = (matT[self.data_class[j]].shape[2] + matE[self.data_class[j]].shape[2])
            if label_flag[i] > 0:
                labels.extend([j] * total_l)
            else:
                labels.extend([-1] * total_l)
            test_labels.extend([j] * total_l)
            ids.extend([s] * total_l) 
        subject_signal = torch.cat(subject_signal)
        return subject_signal, labels, test_labels, ids
    

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


