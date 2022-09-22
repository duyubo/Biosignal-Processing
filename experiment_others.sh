batch_size=256
ratio_list=(0.01 0.05 0.1 0.2 0.3 0.4 0.6 0.8)
lebeled_list=(0.01 0.1 0.2 0.4 0.6 0.8 1)
methods_list="SimCLR WCL CLOCS"
labeled_data=false
labeled_subjects=false
subjects=true
supervised=true
#sota methods: labeled data and labeled subjects
for method in $methods_list; do
	if $labeled_data || $labeled_subjects; then
	echo "deal with sota methods"
	python3 ./unsupervised.py \
			--method $method --backbone CNN --lr 0.001 --epochs 200 --dataset EEG109\
			--dataset_path ../BiosignalData/eeg_109_imagery.npy\
			--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 --stride 2 --first_kernel 5 \
			--first_stride 2 --mlp_hidden_size 128 --projection_size 64  --predictor_mlp_hidden_size 512 \
			--patience 10 --temperature 0.1\
			--train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --batch_size $batch_size \
			--final_dim 4 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2
	fi
	if $labeled_subjects; then
	for train_ratio in "${ratio_list[@]}"; do
		python3 ./supervised.py \
          		--method $method --backbone CNN --lr 0.0001 --epochs 1000 --dataset EEG109 \
          		--dataset_path ../BiosignalData/eeg_109_imagery.npy\
          		--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          		--stride 2 --first_kernel 5 --labeled_data 1\
          		--first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
          		--test_ratio 0.1 --final_dim 4 --pretrained
	done
	fi
	if $labeled_data; then
	for labeled_data in "${lebeled_list[@]}"; do
		python3 ./supervised.py \
			--method $method --backbone CNN --lr 0.0001 --epochs 1000 --dataset EEG109 \
			--dataset_path ../BiosignalData/eeg_109_imagery.npy\
			--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 \
			--stride 2 --first_kernel 5 --labeled_data $labeled_data\
			--first_stride 2 --patience 100 --train_ratio 0.8 --val_ratio 0.1 \
			--test_ratio 0.1 --final_dim 4 --pretrained
	done
	fi
done
#sota subjects
if $subjects; then
echo "deal with number of subjects of sota"
for method in $methods_list; do
	for train_ratio in "${ratio_list[@]}"; do
		python3 ./unsupervised.py \
			--method $method --backbone CNN --lr 0.001 --epochs 200 --dataset EEG109\
			--dataset_path ../BiosignalData/eeg_109_imagery.npy\
			--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 --stride 2 --first_kernel 5 \
			--first_stride 2 --mlp_hidden_size 128 --projection_size 64  --predictor_mlp_hidden_size 512 \
			--patience 10 --temperature 0.1\
			--train_ratio $train_ratio --val_ratio 0.1 --test_ratio 0.1 --batch_size $batch_size \
			--final_dim 4 --labeled_data 1 --p1 0.2 --p2 0.2 --p3 0.2
	
		python3 ./supervised.py \
          		--method $method --backbone CNN --lr 0.0001 --epochs 1000 --dataset EEG109 \
          		--dataset_path ../BiosignalData/eeg_109_imagery.npy\
          		--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          		--stride 2 --first_kernel 5 --labeled_data 1\
          		--first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
          		--test_ratio 0.1 --final_dim 4 --pretrained
	done
done
fi
#supervised learning
if $supervised; then
	echo "deal with supervised"
	for labeled_data in "${lebeled_list[@]}"; do
		python3 ./supervised.py \
		--method PSL --backbone CNN --lr 0.0001 --epochs 1000 --dataset EEG109 \
		--dataset_path ../BiosignalData/eeg_109_imagery.npy\
		--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5 \
		--stride 2 --first_kernel 5 --labeled_data $labeled_data\
		--first_stride 2 --patience 100 --train_ratio 0.8 --val_ratio 0.1 \
		--test_ratio 0.1 --final_dim 4
	done
	for train_ratio in "${ratio_list[@]}"; do
		python3 ./supervised.py \
          	--method PSL --backbone CNN --lr 0.0001 --epochs 1000 --dataset EEG109 \
          	--dataset_path ../BiosignalData/eeg_109_imagery.npy\
          	--seq_length 646 --input_dim 64 --c2 128 --c3 128 --out_dim 64 --kernel 5\
          	--stride 2 --first_kernel 5 --labeled_data 1\
          	--first_stride 2 --patience 100 --train_ratio $train_ratio --val_ratio 0.1 \
        	--test_ratio 0.1 --final_dim 4
	done
fi


