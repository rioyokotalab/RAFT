#!/bin/bash

START_TIMESTAMP=$(date '+%s')

git_root=$(git rev-parse --show-toplevel | head -1)

. /etc/profile.d/modules.sh

module load cuda/11.1

data_root=$(
    cd "$1" || exit
    pwd
)
flow_out="$data_root/flow"
script_name=$(basename "$0")
script_dir="$data_root/flow"
mkdir -p "$script_dir"

pushd "$git_root"

python flow_save_bdd100k.py \
    --output "$data_root/flow" \
    --root "$data_root" \
    --subset "train" \
    --start 0 \
    --datanum 10 \
    --format-save "torch_save" \
    --model "$git_root/models/raft-small.pth" \
    --small \
    --mixed_precision \
    --iters 20 \
    --warm-start \
    --time-offprint \
    # --localtime-offprint \
    # --all-offprint \
    # --debug \
    # --all \
    # --random \
    # --format-save "pickle" \
    # --model "$git_root/models/raft-things.pth" \
    # --start
    # --alternate_corr \

popd

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo $E_TIME

cp "$script_name" "$script_dir"
