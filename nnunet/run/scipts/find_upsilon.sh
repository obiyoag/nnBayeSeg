data_base="/home/gaoyibo/Datasets/nnUNet_datasets/nnUNet_raw_data_base/nnUNet_raw_data"
phi_upsilon=-10
task_id=509
gpu_id=2

task_name="Task${task_id}_phi_upsilon=1e${phi_upsilon}"
mkdir -p "${data_base}/${task_name}/predictions"
cp -r "${nnUNet_preprocessed}/Task500_FeTA_1" "${nnUNet_preprocessed}/${task_name}"

phi_upsilon=`echo "10^${phi_upsilon}" | bc -l`
CUDA_VISIBLE_DEVICES=${gpu_id} python -u ../new_run_training.py 2d BayeSegTrainer ${task_name} 0 --npz --batch_size 24 --phi_upsilon ${phi_upsilon}

input_folder="${data_base}/Task501_FeTA_2/imagesTr"
output_folder="${data_base}/${task_name}/predictions"
CUDA_VISIBLE_DEVICES=${gpu_id} nnUNet_predict -i ${input_folder} -o ${output_folder} -t ${task_name} -tr BayeSegTrainer -m 2d -f 0

ref_folder="${data_base}/Task501_FeTA_2/labelsTr"
nnUNet_evaluate_folder -ref ${ref_folder} -pred ${output_folder} -l 1 2 3 4 5 6 7

rm -rf "${nnUNet_preprocessed}/${task_name}"
