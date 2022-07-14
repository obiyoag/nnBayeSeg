data_base="/home/gaoyibo/Datasets/nnUNet_datasets/nnUNet_raw_data_base/nnUNet_raw_data"

task_name="Task501_FeTA_2"
CUDA_VISIBLE_DEVICES=1 python -u ../new_run_training.py 2d BayeSegTrainer ${task_name} 0 --npz --batch_size 56

input_folder="${data_base}/Task500_FeTA_1/imagesTr"
output_folder="${data_base}/${task_name}/predictions"
CUDA_VISIBLE_DEVICES=1 nnUNet_predict -i ${input_folder} -o ${output_folder} -t ${task_name} -tr BayeSegTrainer -m 2d -f 0

ref_folder="${data_base}/Task500_FeTA_1/labelsTr"
nnUNet_evaluate_folder -ref ${ref_folder} -pred ${output_folder} -l 1 2 3 4 5 6 7
