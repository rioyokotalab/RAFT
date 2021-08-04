#!/bin/bash
cur_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
git_root=$(git rev-parse --show-toplevel | head -1)

job_dir="$git_root/jobs/gen_flow"

datanum=350
all_train_datanum=70000
file_train_num=$((($all_train_datanum + $datanum - 1) / $datanum))
all_val_datanum=10000
file_val_num=$((($all_val_datanum + $datanum - 1) / $datanum))

bash "$cur_dir"/job_sub.sh "$job_dir" 1 $file_train_num "train"
bash "$cur_dir"/job_sub.sh "$job_dir" 1 $file_val_num "val"
