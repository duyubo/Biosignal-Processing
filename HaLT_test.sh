python3 ./supervised.py \
	--method SimCLR --backbone CNN --lr 0.001 --epochs 1000 --dataset HaLT12\
	--dataset_path /home/yubo/BiosignalData/HaLT12.npy\
	--seq_length 200 --input_dim 22 --c2 48 --c3 64 --out_dim 48 --kernel 5 --stride 1 --first_kernel 5 \
	--first_stride 2 --patience 100\
	--train_ratio 0.95 --val_ratio 0.2 --test_ratio 0.05 --batch_size 128\
	--final_dim 5 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2 --pretrained --seed 37

:"python3 ./supervised.py \
	--method PSL --backbone CNN --lr 0.001 --epochs 1000 --dataset HaLT12\
	--dataset_path /home/yubo/BiosignalData/HaLT12.npy\
	--seq_length 200 --input_dim 22 --c2 48 --c3 64 --out_dim 48 --kernel 5 --stride 1 --first_kernel 5 \
	--first_stride 2 --patience 100\
	--train_ratio 0.95 --val_ratio 0.2 --test_ratio 0.05 --batch_size 128\
	--final_dim 5 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2  --seed 37"

