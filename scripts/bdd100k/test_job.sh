#!/bin/bash

START_TIMESTAMP=$(date '+%s')

git_root=$(git rev-parse --show-toplevel | head -1)

. /etc/profile.d/modules.sh

module load cuda/11.1

data_root=$(
    cd "$1" || exit
    pwd
)

pushd "$git_root"

python flow_save_bdd100k.py \
    --output "result" \
    --root "$data_root" \
    --subset "train" \
    --debug \
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

