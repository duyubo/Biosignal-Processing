import os
from glob import glob
import numpy as np


def load_folds_data(np_data_path, n_folds):
    """
    Input: 
    data path of .npz files
    k folder number
    Output:
    file names group by subject id, split by folds
    folds_data = {0: [training file names][test file names],..., k-1:[training file names][test file names]}

    Derive from : https://github.com/emadeldeen24/AttnSleep/blob/37eb7acf59bde9cb979a2665d5331277e0a60b69/utils/util.py
    AttnSleep: An Attention-based Deep Learning Approach for Sleep Stage Classification with Single-Channel EEG
    """
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_permute = np.array([42, 11, 19, 46, 24, 47, 61, 71, 58, 66,  6, 22, 31,  5, 12, 54,  4, 67, 37, 68, 48, 39, 13, 15,
    20, 23, 70, 18,  3, 64, 55, 62, 27,  7,  0, 36, 40, 44, 50,  2, 75, 43, 26,  1, 29, 14, 72, 57,
    73, 59, 32, 65, 34, 63, 69, 33, 35, 77, 53, 16, 74, 56, 28, 17, 52, 30, 76, 51, 49, 25, 21, 38,
    8, 41,  9, 10, 60, 45])
    
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs)
    files_pairs = files_pairs[r_permute]

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data


"""Test"""
if __name__ == "__main__":
    load_folds_data('/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_EDF78/EEG_Sleep_Processed', 5)