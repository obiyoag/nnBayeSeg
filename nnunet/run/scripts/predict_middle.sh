#!/usr/bin/env bash

gpu_id=1
test_data_name="Task500_FeTA_1"
task_name="Task508_weight50.0_on2"

data_base="/home/gaoyibo/Datasets/nnUNet_datasets/nnUNet_raw_data_base/nnUNet_raw_data"

image_folder="${data_base}/${test_data_name}/imagesTr"
label_folder="${data_base}/${test_data_name}/labelsTr"
pred_folder="${data_base}/${task_name}/predictions"

from_folder="${RESULTS_FOLDER}/nnUNet/2d/${task_name}/BayeSegTrainer__nnUNetPlansv2.1/fold_0"
to_folder="$HOME/FeTA_results/${task_name}"

cp "${from_folder}/model_best.model" "${from_folder}/model_final_checkpoint.model"
cp "${from_folder}/model_best.model.pkl" "${from_folder}/model_final_checkpoint.model.pkl"

CUDA_VISIBLE_DEVICES=${gpu_id} nnUNet_predict -i ${image_folder} -o ${pred_folder} -t ${task_name} -tr BayeSegTrainer -m 2d -f 0
nnUNet_evaluate_folder -ref ${label_folder} -pred ${pred_folder} -l 1 2 3 4 5 6 7

mkdir -p "${to_folder}"
mv "${from_folder}/progress.png" "${to_folder}/progress.png"
mv "${pred_folder}/summary.json" "${to_folder}/test.json"
rm -rf ${pred_folder}
