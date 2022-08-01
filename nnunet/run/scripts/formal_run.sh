#!/usr/bin/env bash

gpu_id=3
fold_id="all"
task_name="Task502_FeTA_both"

CUDA_VISIBLE_DEVICES=${gpu_id} python -u ../new_run_training.py 2d BayeSegTrainer ${task_name} ${fold_id} --npz
