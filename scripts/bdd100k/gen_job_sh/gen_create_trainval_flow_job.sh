#!/bin/bash
script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
git_root=$(git rev-parse --show-toplevel | head -1)

tar_script_name="$git_root/scripts/bdd100k/create_flow.sh"
out_dir="$git_root/jobs/gen_flow"
mkdir -p "$out_dir"
result_dir="$git_root/output/"

data_root="$1"
datanum=350
all_train_datanum=70000
all_val_datanum=10000
file_train_num=$((($all_train_datanum + $datanum - 1) / $datanum))
file_val_num=$((($all_val_datanum + $datanum - 1) / $datanum))

bash "$script_dir"/gen_create_flow_job.sh "$tar_script_name" "$out_dir" "$data_root" "$result_dir" $file_train_num "$datanum" "train"
bash "$script_dir"/gen_create_flow_job.sh "$tar_script_name" "$out_dir" "$data_root" "$result_dir" $file_val_num "$datanum" "val"
