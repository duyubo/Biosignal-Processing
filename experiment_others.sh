batch_size=256
ratio_list=(0.01 0.05 0.1 0.2 0.4 0.6 0.8)
lebeled_list=(0.01 0.1 0.2 0.4 0.8 1)
methods_list="SimCLR WCL CLOCS"
for method in $methods_list; do
	for train_ratio in "${ratio_list[@]}"; do
		python3 ./supervised.py \
          		--method $method --backbone CNN --lr 0.0001 --epochs 200 --dataset EEG109 \
          		--dataset_path ../eeg_109_imagery.npy\
          		--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          		--stride 2 --first_kernel 5 --labeled_data 1\
          		--first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
          		--test_ratio 0.1 --final_dim 4 --pretrained
	done

	for labeled_data in "${lebeled_list[@]}"; do
		python3 ./supervised.py \
			--method $method --backbone CNN --lr 0.0001 --epochs 200 --dataset EEG109 \
			--dataset_path ../eeg_109_imagery.npy\
			--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
			--stride 2 --first_kernel 5 --labeled_data $labeled_data\
			--first_stride 2 --patience 100 --train_ratio 0.8 --val_ratio 0.1 \
			--test_ratio 0.1 --final_dim 4 --pretrained
	done
done

