#!/bin/bash

START_TIMESTAMP=$(date '+%s')

git_root=$(git rev-parse --show-toplevel | head -1)
htime_bash="$git_root/scripts/bdd100k/measure_time/change_humantime.sh"

. /etc/profile.d/modules.sh

# module load cuda/11.1
module load cuda/10.1.105

source "$HOME"/.bashrc


data_root=$(
    cd "$1" || exit
    pwd
)
start_index=$2
data_num=$3
subset="$4"
format_save="$5"
n_frames=$6

flow_model="$7"

output_path="$8"
normalize="$9"

log_file="$JOB_NAME.o$JOB_ID"

flow_out="$output_path"

pushd "$git_root"

if [ -n "$normalize" ]; then
    python myraft_demo.py \
        --output "$flow_out" \
        --root "$data_root" \
        --subset "$subset" \
        --start $start_index \
        --datanum $data_num \
        --format-save "$format_save" \
        --model "$flow_model" \
        --small \
        --mixed_precision \
        --iters 12 \
        --normalize \
        --n_frames $n_frames \
        --warm-start \
        --debug \
        # --time-offprint \
        # --all-offprint \
        # --alternate_corr \
        # --mixed_precision \
else
    python myraft_demo.py \
        --output "$flow_out" \
        --root "$data_root" \
        --subset "$subset" \
        --start $start_index \
        --datanum $data_num \
        --format-save "$format_save" \
        --model "$flow_model" \
        --small \
        --mixed_precision \
        --iters 12 \
        --n_frames $n_frames \
        --warm-start \
        --debug \
        # --normalize \
        # --time-offprint \
        # --all-offprint \
        # --alternate_corr \
        # --mixed_precision \
fi


popd

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
# echo $E_TIME
bash "$htime_bash" $E_TIME "$normalize $n_frames exec time"

# output="output/n_frames_$n_frames/demo_flow"
# mkdir -p "$output"
# cp "$log_file" "$output"