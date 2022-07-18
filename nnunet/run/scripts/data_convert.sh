#!/usr/bin/env bash

# FeTA converting
# task_name="Task502_FeTA_both"
# task_id="502"
# institution="both" # both FeTA2022_Institution1 or FeTA2022_Institution2

# python -u ../data_convert.py --task_name ${task_name} --institution ${institution}
# nnUNet_plan_and_preprocess -t ${task_id}


# ACDC converting
task_name="Task513_ACDC"
task_id="513"

python -u ../ACDC_convert.py --task_name ${task_name}
nnUNet_plan_and_preprocess -t ${task_id}
