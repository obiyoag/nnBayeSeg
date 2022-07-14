PHI_UPSILON="-6 -7 -9 -10"
task_id=502
data_base="/home/gaoyibo/Datasets/nnUNet_datasets/nnUNet_raw_data_base/nnUNet_raw_data"
for phi_upsilon in $PHI_UPSILON
do
	task_name="Task${task_id}_phi_upsilon=1e${phi_upsilon}"
	phi_upsilon=`echo "10^${phi_upsilon}" | bc -l`

	python -u ../data_convert.py --task_name ${task_name}
	nnUNet_plan_and_preprocess -t ${task_id}

	CUDA_VISIBLE_DEVICES=0 python -u ../new_run_training.py 2d BayeSegTrainer ${task_name} 0 --npz --batch_size 56 --phi_upsilon ${phi_upsilon}

	input_folder="${data_base}/Task501_FeTA_2/imagesTr"
	output_folder="${data_base}/${task_name}/predictions"
	CUDA_VISIBLE_DEVICES=0 nnUNet_predict -i ${input_folder} -o ${output_folder} -t ${task_name} -tr BayeSegTrainer -m 2d -f 0

	ref_folder="${data_base}/Task501_FeTA_2/labelsTr"
	nnUNet_evaluate_folder -ref ${ref_folder} -pred ${output_folder} -l 1 2 3 4 5 6 7
	task_id=`echo "${task_id}+1" | bc`

done
