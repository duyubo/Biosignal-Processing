# Biosignal-Processing

## Environment Setup
All necessary envronment are included in environment.yml, after installing anaconda3 on the linux, 
```
conda env create -f environment.yml 
```
will help install all dependencies.

## Run Experiments
```
python3 unsupervised.py \
          --method PSL --backbone CNN --lr 0.001 --epochs 200 --dataset EEG109\
          --dataset_path /home/yubo/BiosignalData/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 --stride 2 --first_kernel 5 \
          --first_stride 2 --mlp_hidden_size 128 --projection_size 64  --predictor_mlp_hidden_size 512 \
          --patience 10 --temperature 0.1\
          --train_ratio 0.8 --train_ratio_all 0.8 --val_ratio 0.1 --test_ratio 0.1 --batch_size 128 \
          --final_dim 4 --labeled_data_all 0.8 --labeled_data $labeled_data --p1 0.2 --p2 0.2 --p3 0.2 --la 1
  python3 supervised.py \
          --method PSL --backbone CNN --lr 0.0001 --epochs 1000 --dataset EEG109 \
          --dataset_path /home/yubo/BiosignalData/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          --stride 2 --first_kernel 5 --labeled_data $labeled_data \
          --first_stride 2 --patience 100 --train_ratio 0.8 --val_ratio 0.1 \
          --test_ratio 0.1 --final_dim 4 --pretrained
```


```train_ratio:``` percent of labeled subjects

```train_ratio_all:``` percent of subjects

```labeled_data:``` percent of labeled data per subject

```labeled_data_all:``` percent of data per subject 

## Details of Implementations
  - Backbone/
      - CNN_Backbone.py: Contrastive learning backbone of CNN model
      - Transformer_Backbone.py: Contrastive learning backbone of Transformer model
  - Baselines/
      - SimCLR.py: The most classic contrastive learning approach [2] 
      - BYOL.py: The first contrastive learning framework with two neural networks, referred to as online and target networks, that interact and learn from each other [1]
      - MocoV3.py: A recent contrasntive learning framework with two neural networks with ViT backbones [3]
      - CLOCS.py: The most classic contrasntive learning framework for biosignal processing [4]
      - MAE.py: An auto-regression self-supervised learning framework that learns how to reconstruct the masked input information [5]
      - WCL.py: Weakly-supervied contrastive learning that boosts the contrastive learning process by guessing some pseudo labels [6]
      - PSL.py: Pairwise supervised contrsative learning proposed by us (more details are included in the following sections)
  - Preprocess_Data: raw data preprocessing for different datasets
      - load_chapman_ecg.py / read_HaLT12.py / read_edf78.py / read_eeg109.py / read_emg22.py / read_ofner.py
  - augmentation.py: Candidate data augmentation approaches
  - dataset.py: Dataloaders for each dataset (supervised learning of PSL and supervised & unsupervised learning of the other baseline methods)
  - dataset_pretrain.py: Dataloader for each dataset (unsupervised learning of PSL)
  - supervised.py: Supervised learning framework (that calls training functions in train.py)
  - train.py: Training functions for  un/supervised training, un/supervised evaluation.
  - unsupervised.py: Unsupervised learning framework (that calls training functions in train.py)
  - utils.py: Utilization functions
  - v.py: Visualization fuctions for an intuitive observation of whether the constrastive learning backbones can distinguish the differences between different classes

## Datasets
- EEG109: 64 channel eeg data of 109 subjects recording motor imagery and motor executation [9][10][11] 
- BCI-IV 2a: 22 channel eeg data of 9 subjects recording motor imagery [14]
- EDF78: single channel eeg data of 78 subjects recording eeg signals while the people are sleeping [11][12][13]
- EMG22: 12-channel emg signals of 22 subjects recording basic movements of the fingers and of the wrist and grasping and functional movements [7][8]
- Cho2017: 64-channel eeg signals of 52 subjects recording motor imagery of the left and right hands [15][17]

## Motivation && Problem Defination
With some preliminary experiment results, we find that the biosignals such as EEGs of motor imagery are different among subjects. These differences bring the diffculties in cross-subject biosignal classification tasks. For example, if we train the DL model on first n subjects (each subject k data samples) while test the remaining m subjects (each subject k data samples), we will have much lower classification accuracy than training on m+n subjects (each subject has k1 data samples)
and test on the same m+n subjects but different data samples (each subject has k2 data samples, k1 + k2 = k). 

Although collecting biosignals from a large amount of subjects can help reduce this gap, the cost in time and money are expensive. Most datasets only contain 20 to 40 subjects in total. 

Based on theses observations, we want to explore if we can have a data efficient method that enables the model to learn the essential information from the given subjects and can be applied to unknown subjects. Contrastive learning is a popular approach to help DL models with less labeled data to achieve better performance. However, the previous contrastive learning methods only focus on number of labeled data per class. The cost of looking for & training subjects and label for signals are both non-trivial. Then we propose three experiment settings for our contrastive learning approach:

If there are m+n subjects in total, all training & validation data comes from first n subjects and test data from m subjects. There are no overlapping of subjects between train/val and test.
- **Labeled Data Per Subject vs Accuracy**: Variation in the number of labeled data per subjects versus accuracy change
- **Labeled Subjects vs Accuracy**: Variation in the number of labeled subjects versus accuracy change 
- **Subjects vs Accuracy**: Variation in the number of subjects versus accuracy change, here all avaliable subjects are labeled and there are no unlabeled subjects can be used

## Solution
There are two key points in our solution:
### 1. Label reorganization
![](data_reorganize.JPG)
### 2. Semi-supervised learning with both labeled & unlabeled data
![](unlabeled_entropy.JPG)

## Current Status 
### Sota accuracy on EEG109 dataset
![](result1.JPG)![](result2.JPG)![](result3.JPG)

## Next Steps
### 1. Look for more datasets
PSL currently only has sota performance on EEG109 datasets and does not have much differences between the other contrastive learning approaches on other datasets.
### 2. Explore the effectiveness of the transformer based pairwise modules (Theory/Experiment)
We should figure out why the pairwise difference modules help on EEG19 dataset. We can do this either from mathematic side or ablation study
### 3. Add more baselines (contrastive learning for biosignal processing)
We should also add more baselines methods that expeciaffically proposed for biosignal processing. Here are some candidate baselines:
**Here is a collection of related works/baselines**
- General contrastive learning. (SimCLR[2], BYOL[1])
- Supervised contrastive learning (SCL)
- Contrastive learning for general time series. (TPC[19], TFC[18])
- Contrastive learning for bio-signals considering variances in subjects.  (BENDR[24], CLOCS[4], SACL[25])


## References
[1] Grill, Jean-Bastien, et al. "Bootstrap your own latent-a new approach to self-supervised learning." Advances in neural information processing systems 33 (2020): 21271-21284.

[2] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.

[3] Chen, Xinlei, Saining Xie, and Kaiming He. "An empirical study of training self-supervised vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[4] Kiyasseh, Dani, Tingting Zhu, and David A. Clifton. "Clocs: Contrastive learning of cardiac signals across space, time, and patients." International Conference on Machine Learning. PMLR, 2021.

[5] He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[6] Zheng, Mingkai, et al. "Weakly supervised contrastive learning." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[7] Krasoulis, Agamemnon, et al. "Improved prosthetic hand control with concurrent use of myoelectric and inertial measurements." Journal of neuroengineering and rehabilitation 14.1 (2017): 1-14.

[8] http://ninapro.hevs.ch/DB7_Instructions

[9] https://www.physionet.org/content/eegmmidb/1.0.0/

[10] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

[11] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

[12] https://physionet.org/content/sleep-edfx/1.0.0/

[13] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE-BME 47(9):1185-1194 (2000).

[14] https://www.bbci.de/competition/iv/desc_2a.pdf

[15] Cho, Hohyun, et al. "EEG datasets for motor imagery brain–computer interface." GigaScience 6.7 (2017): gix034.

[16] Shin, Jaeyoung, et al. "Open access dataset for EEG+ NIRS single-trial classification." IEEE Transactions on Neural Systems and Rehabilitation Engineering 25.10 (2016): 1735-1745.

[17] http://moabb.neurotechx.com/docs/index.html

[18] TFC: Zhang, Xiang, et al. "Self-supervised contrastive pre-training for time series via time-frequency consistency." arXiv preprint arXiv:2206.08496 (2022).
CLUDA: Ozyurt, Yilmazcan, Stefan Feuerriegel, and Ce Zhang. "Contrastive Learning for Unsupervised Domain Adaptation of Time Series." arXiv preprint arXiv:2206.06243 (2022).

[19] TPC:Tonekaboni, Sana, Danny Eytan, and Anna Goldenberg. "Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding." International Conference on Learning Representations. 2020.

[20] Emadeldeen Eldele et al. “Time-Series Representation Learning via Temporal and Contextual Contrasting”. In: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21. 2021, pp. 2352–2359.

[21] Yue, Zhihan, et al. "Ts2vec: Towards universal representation of time series." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.

[22]Mohsenvand, Mostafa Neo, Mohammad Rasool Izadi, and Pattie Maes. "Contrastive representation learning for electroencephalogram classification." Machine Learning for Health. PMLR, 2020.

[23]Han, Jinpei, Xiao Gu, and Benny Lo. "Semi-supervised contrastive learning for generalizable motor imagery eeg classification." 2021 IEEE 17th International Conference on Wearable and Implantable Body Sensor Networks (BSN). IEEE, 2021.

[24]Kostas, Demetres, Stephane Aroca-Ouellette, and Frank Rudzicz. "BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data." Frontiers in Human Neuroscience (2021): 253.

[25]Cheng, Joseph Y., et al. "Subject-aware contrastive learning for biosignals." arXiv preprint arXiv:2007.04871 (2020).
