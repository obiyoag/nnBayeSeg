train_data_name="Task502_FeTA_both"
gpu_id=2
fold_id=4
task_name="both_fold4"

cp -r "${nnUNet_preprocessed}/${train_data_name}" "${nnUNet_preprocessed}/${task_name}"
CUDA_VISIBLE_DEVICES=${gpu_id} python -u ../new_run_training.py 2d BayeSegTrainer ${task_name} ${fold_id} --npz

