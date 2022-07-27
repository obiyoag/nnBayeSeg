#!/usr/bin/env bash

gpu_id=1
task_id=504
lbd=0.6
hbd=0.9
weight=100.0
task_name="Task${task_id}_range_${lbd}_${hbd}_weight${weight}_2d_ddo_clipx"
train_data_name="Task500_FeTA_1"
test_data_name="Task501_FeTA_2"

data_base="/home/gaoyibo/Datasets/nnUNet_datasets/nnUNet_raw_data_base/nnUNet_raw_data"

mkdir -p "${data_base}/${task_name}/predictions"

image_folder="${data_base}/${test_data_name}/imagesTr"
label_folder="${data_base}/${test_data_name}/labelsTr"
pred_folder="${data_base}/${task_name}/predictions"

from_folder="${RESULTS_FOLDER}/nnUNet/2d/${task_name}/BayeSegTrainer__nnUNetPlansv2.1/fold_0"
to_folder="$HOME/FeTA_results/${task_name}"

CUDA_VISIBLE_DEVICES=${gpu_id} nnUNet_predict -i ${image_folder} -o ${pred_folder} -t ${task_name} -tr BayeSegTrainer -m 2d -f 0
nnUNet_evaluate_folder -ref ${label_folder} -pred ${pred_folder} -l 1 2 3 4 5 6 7

mkdir -p "${to_folder}"
mv "${pred_folder}/summary.json" "${to_folder}/test.json"
rm -rf "${nnUNet_preprocessed}/${task_name:?}"
rm -rf "${pred_folder}"
