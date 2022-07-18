#!/usr/bin/env bash

task_name="Task502_FeTA_both"
task_id="502"
institution="both" # both FeTA2022_Institution1 or FeTA2022_Institution2

python -u ../data_convert.py --task_name ${task_name} --institution ${institution}
nnUNet_plan_and_preprocess -t ${task_id}
