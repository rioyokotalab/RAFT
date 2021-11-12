#!/bin/bash
cur_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
git_root=$(git rev-parse --show-toplevel | head -1)

bdd100k_root=$(
    cd "$1" || exit
    pwd
)
bdd100k_root_path="$bdd100k_root/bdd100k"
bdd100k_flow_script_path="$bdd100k_root_path/flow/scripts"
date_str=$(date '+%Y%m%d_%H%M%S')

pushd "$git_root"
git_out="$bdd100k_flow_script_path/git_out/$date_str"
mkdir -p "$git_out"
git status > "$git_out/git_status.txt"
git log > "$git_out/git_log.txt"
git diff HEAD > "$git_out/git_diff.txt"
git rev-parse HEAD > "$git_out/git_head.txt"
popd

job_dir="$git_root/jobs/gen_flow"

datanum=350
all_train_datanum=70000
file_train_num=$((($all_train_datanum + $datanum - 1) / $datanum))
all_val_datanum=10000
file_val_num=$((($all_val_datanum + $datanum - 1) / $datanum))

bash "$cur_dir"/job_sub.sh "$job_dir" 1 $file_train_num "train"
bash "$cur_dir"/job_sub.sh "$job_dir" 1 $file_val_num "val"
