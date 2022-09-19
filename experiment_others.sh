batch_size=256
ratio_list=(0.01 0.05 0.1 0.2 0.3 0.4 0.6 0.8)
for train_ratio in "${ratio_list[@]}"; do
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/unsupervised.py \
          --method CLOCS --backbone CNN --lr 0.001 --epochs 200 --dataset EEG109\
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 --stride 2 --first_kernel 5 \
          --first_stride 2 --mlp_hidden_size 128 --projection_size 64  --predictor_mlp_hidden_size 512 \
          --patience 10 --temperature 0.1\
          --train_ratio $train_ratio --val_ratio 0.1 --test_ratio 0.1 --batch_size $batch_size \
          --final_dim 4 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/supervised.py \
          --method CLOCS --backbone CNN --lr 0.0001 --epochs 200 --dataset EEG109 \
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          --stride 2 --first_kernel 5 --labeled_data 1\
          --first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
          --test_ratio 0.1 --final_dim 4 --pretrained
done
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/unsupervised.py \
          --method SimCLR --backbone CNN --lr 0.001 --epochs 200 --dataset EEG109\
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 --stride 2 --first_kernel 5 \
          --first_stride 2 --mlp_hidden_size 128 --projection_size 64  --predictor_mlp_hidden_size 512 \
          --patience 10 --temperature 0.1\
          --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --batch_size $batch_size \
          --final_dim 4 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2
ratio_list=(0.01 0.1 0.2 0.4 0.6 0.8)
lebeled_list=(0.01 0.1 0.2 0.4 0.8 1)
for train_ratio in "${ratio_list[@]}"; do
for labeled_data in "${lebeled_list[@]}"; do
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/supervised.py \
          --method SimCLR --backbone CNN --lr 0.0001 --epochs 200 --dataset EEG109 \
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          --stride 2 --first_kernel 5 --labeled_data $labeled_data\
          --first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
          --test_ratio 0.1 --final_dim 4 --pretrained
done
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/unsupervised.py \
          --method WCL --backbone CNN --lr 0.001 --epochs 200 --dataset EEG109\
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 --stride 2 --first_kernel 5 \
          --first_stride 2 --mlp_hidden_size 128 --projection_size 64  --predictor_mlp_hidden_size 512 \
          --patience 10 --temperature 0.1\
          --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --batch_size $batch_size \
          --final_dim 4 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2
ratio_list=(0.01 0.1 0.2 0.4 0.6 0.8)
lebeled_list=(0.01 0.1 0.2 0.4 0.8 1)
for train_ratio in "${ratio_list[@]}"; do
for labeled_data in "${lebeled_list[@]}"; do
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/supervised.py \
          --method WCL --backbone CNN --lr 0.0001 --epochs 200 --dataset EEG109 \
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          --stride 2 --first_kernel 5 --labeled_data $labeled_data\
          --first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
          --test_ratio 0.1 --final_dim 4 --pretrained
done
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/unsupervised.py \
          --method CLOCS --backbone CNN --lr 0.001 --epochs 200 --dataset EEG109\
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 --stride 2 --first_kernel 5 \
          --first_stride 2 --mlp_hidden_size 128 --projection_size 64  --predictor_mlp_hidden_size 512 \
          --patience 10 --temperature 0.1\
          --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --batch_size $batch_size \
          --final_dim 4 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2
ratio_list=(0.01 0.1 0.2 0.4 0.6 0.8)
lebeled_list=(0.01 0.1 0.2 0.4 0.8 1)
for train_ratio in "${ratio_list[@]}"; do
for labeled_data in "${lebeled_list[@]}"; do
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/supervised.py \
          --method CLOCS --backbone CNN --lr 0.0001 --epochs 200 --dataset EEG109 \
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          --stride 2 --first_kernel 5 --labeled_data $labeled_data\
          --first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
          --test_ratio 0.1 --final_dim 4 --pretrained
done
lebeled_list=(0.01 0.1 0.2 0.4 0.8 1)
for labeled_data in "${lebeled_list[@]}"; do
python3 /content/drive/MyDrive/Colab_Notebooks/Contrastive_Learning_Biosignal/supervised.py \
          --method CLOCS --backbone CNN --lr 0.0001 --epochs 200 --dataset EEG109 \
          --dataset_path /content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data/EEG_Motery_and_imagery/eeg-motor-movementimagery-dataset-1/eeg_109_imagery.npy\
          --seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          --stride 2 --first_kernel 5 --labeled_data $labeled_data\
          --first_stride 2 --patience 100 --train_ratio 0.8 --val_ratio 0.1 \
          --test_ratio 0.1 --final_dim 4
done