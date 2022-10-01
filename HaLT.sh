python3 ./unsupervised.py \
  --method PSL --backbone CNN --lr 0.001 --epochs 1000 --dataset HaLT12\
  --dataset_path /home/yubo/BiosignalData/HaLT12.npy\
  --seq_length 200 --input_dim 22 --c2 48 --c3 64 --out_dim 48 --kernel 3 --stride 1 --first_kernel 5 \
  --first_stride 2 --mlp_hidden_size 64 --projection_size 48  --predictor_mlp_hidden_size 48 \
  --patience 3 --temperature 0.1\
  --train_ratio_all 0.95 --train_ratio 0.95 --data_ratio_train 0.001 --data_ratio_val 0.001\
  --val_ratio 0.2 --test_ratio 0.05 --batch_size 48\
  --final_dim 5 --labeled_data_all 1 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2 --topk 1 --la 0.1 --seed 121
