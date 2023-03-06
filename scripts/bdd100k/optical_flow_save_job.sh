#!/bin/bash

#------ pjsub option --------#
#PJM -g jhpcn2628
#PJM -L node=64
#PJM --mpi proc=256
#PJM -L rscgrp=cx-large
#PJM -L elapse=72:00:00
#PJM -N save_optical_flow
#PJM -j
#PJM -X

set -x

echo "start scirpt file cat"
cat "$0"

set +x

echo "end scirpt file cat"

START_TIMESTAMP=$(date '+%s')
# \#PJM -L rscgrp=cxgfs-special
# \#PJM -L rscunit=cx

# ======== augs ========

raft_name=${RAFT_NAME:-"small"}
subset="train"
data_name="flow"

ext=".pth"
# ext=".flo"
# ext=".png"

s_num=${S_NUM:-0}
make_num=${NUM:-70000}
bdd100k_root=${BDD100k:-"/data/bdd100k/images"}

# ======== Variables ========

job_id_base=$PJM_JOBID

git_root=$(git rev-parse --show-toplevel | head -1)
# base_root=$(basename "$git_root")
data_root="$bdd100k_root"


flow_model="$git_root/models/raft-$raft_name.pth"

raft_opts="--model $flow_model"
if [ "$raft_name" = "small" ];then
    raft_opts+=" --small"
fi

log_file="$PJM_JOBNAME.$job_id_base.out"
date_str=$(date '+%Y%m%d_%H%M%S')

raft_opts+=" --save_type $ext"

raft_opts+=" --subset $subset"


# raft_opts+=" --split_file"
# raft_opts+=" --start_idx $s_num"
# raft_opts+=" --make_num $make_num"

cur_out="$git_root/output/logs/optical_flow_save"
commot_out="$cur_out/$date_str"
git_out="$commot_out/git_out"
script_out="$commot_out/scripts"
# raft_opts+=" --out_root "$cur_out/$date_str"

# ======== Pyenv ========

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# pipenv property
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
# which python

# ======== Modules ========

module load cuda/10.2.89_440.33.01
module load nccl/2.7.3
# module load cuda/11.3.1
# module load nccl/2.9.9

module load gcc/8.4.0
module load cmake/3.21.1
module load cudnn/8.2.1
# module load openmpi_cuda/4.0.4
module load openmpi/4.0.4

module list

# ======== MPI ========

nodes=$PJM_NODE
# gpus_pernode=4
# cpus_pernode=5
gpus_pernode=${PJM_PROC_BY_NODE}
# cpus_pernode=${PJM_PROC_BY_NODE}

gpus=${PJM_MPI_PROC}
# cpus=${PJM_MPI_PROC}
# cpus=$nodes
# cpus=$(($nodes * $cpus_pernode))
# gpus=$(($nodes * $gpus_pernode))

# echo "cpus: $cpus"
# echo "cpus per node $cpus_pernode"

echo "gpus: $gpus"
echo "gpus per node $gpus_pernode"

MASTER_ADDR=$(cat "$PJM_O_NODEINF" | head -1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

MPI_OPTS="-machinefile $PJM_O_NODEINF"
MPI_OPTS+=" -np $gpus"
MPI_OPTS+=" -npernode $gpus_pernode"
MPI_OPTS+=" -x MASTER_ADDR=$MASTER_ADDR"
MPI_OPTS+=" -x MASTER_PORT=$MASTER_PORT"
MPI_OPTS+=" -x NCCL_BUFFSIZE=1048576"
MPI_OPTS+=" -x NCCL_IB_DISABLE=1"
MPI_OPTS+=" -x NCCL_IB_TIMEOUT=14"

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"


# ======== Scripts ========


pushd "$git_root"

set -x

mkdir -p "$git_out"
mkdir -p "$script_out"
script_name="$(basename "$0")"

cat "$0" > "$script_out/$script_name.sh"

git status | tee "$git_out/git_status.txt"
git log > "$git_out/git_log.txt"
git diff HEAD | tee "$git_out/git_diff.txt"
git rev-parse HEAD | tee "$git_out/git_head.txt"

mpirun ${MPI_OPTS} \
    python flow_save_scripts.py \
    --path "$data_root" \
    --data_name "flow" \
    ${raft_opts}

popd

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"

# real_out_path=$(find "$out_path" -name "$date_str" | grep -v "$commot_out" | head -1)
# mv "$git_out" "$real_out_path"
# mv "$script_out" "$real_out_path"
# cp "$log_file" "$real_out_path"
# rmdir "$commot_out"

cp "$log_file" "$commot_out"

