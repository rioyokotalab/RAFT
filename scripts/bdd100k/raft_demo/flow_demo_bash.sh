#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=06:30:00
#$ -N demo_flow
#$ -j y

git_root=$(git rev-parse --show-toplevel | head -1)
script_dir="$git_root/scripts/bdd100k/raft_demo"
# "#\$ -v USE_BEEOND=1

START_TIMESTAMP=$(date '+%s')

htime_bash="$git_root/scripts/bdd100k/measure_time/change_humantime.sh"
git_sh="$git_root/scripts/bdd100k/git_sh/git_log.sh"

bdd100k_root=$(
    cd "$1" || exit
    pwd
)
##########################################################################################
echo "copy to local ssd"

COPY_START_TIMESTAMP=$(date '+%s')
# beegfs-df -p /beeond
# local_dir="$SGE_LOCALDIR"
# local_dir="$SGE_BEEONDDIR"
local_dir="$TMPDIR"
# local_dir="/beeond"
df -hT "$local_dir"

first_image="bdd100k/images/train/0000f77c-6257be58"
mkdir -p "$local_dir/$first_image"

data_dir="$bdd100k_root/$first_image"
rsync -az "$data_dir/" "$local_dir/$first_image"

# pushd "$bdd100k_root"
# find "bdd100k/images/" -type f -name "*.jpg" | parallel --joblog "parallel.log" -j $cpus rsync -azhR {} "$local_dir/"
# popd

ls "$local_dir" -lha
du -d3 -hc --apparent-size "$local_dir"
data_root="$local_dir/bdd100k"

COPY_END_TIMESTAMP=$(date '+%s')

COPY_E_TIME=$(($COPY_END_TIMESTAMP-$COPY_START_TIMESTAMP))
# echo $COPY_E_TIME
bash "$htime_bash" $COPY_E_TIME "copy time"
##########################################################################################

# data_root="$bdd100k_root/bdd100k"
start_index=0
data_num=1
subset="train"
format_save="png"
# format_save="torch_save"
# format_save="pickle"
normalize="normalize"

flow_model="$git_root/models/raft-small.pth"
# flow_model="$git_root/models/raft-things.pth"

date_str=$(date '+%Y%m%d_%H%M%S')
output="output/demo_flow/$date_str"
mkdir -p "$output"
bash "$git_sh" "$output"

common_out="$bdd100k_root/results_root/test_data"

log_file="$JOB_NAME.o$JOB_ID"

raft_script="$script_dir/myraft.sh"


n_frames=2
output_path="$common_out/n_frames_$n_frames"
echo "start $n_frames.."

time bash "$raft_script" "$data_root" $start_index $data_num "$subset" "$format_save" $n_frames "$flow_model" "$output_path"

echo "end $n_frames"

n_frames=2
output_path="$common_out/n_frames_$n_frames/$normalize"
echo "start $normalize $n_frames.."

time bash "$raft_script" "$data_root" $start_index $data_num "$subset" "$format_save" $n_frames "$flow_model" "$output_path" "$normalize"

echo "end $n_frames"

n_frames=10
output_path="$common_out/n_frames_$n_frames"
echo "start $n_frames.."

time bash "$raft_script" "$data_root" $start_index $data_num "$subset" "$format_save" $n_frames "$flow_model" "$output_path"

echo "end $n_frames"

n_frames=10
output_path="$common_out/n_frames_$n_frames/$normalize"
echo "start $normalize $n_frames.."

time bash "$raft_script" "$data_root" $start_index $data_num "$subset" "$format_save" $n_frames "$flow_model" "$output_path" "$normalize"

echo "end $n_frames"

n_frames=15
output_path="$common_out/n_frames_$n_frames"
echo "start $n_frames.."

time bash "$raft_script" "$data_root" $start_index $data_num "$subset" "$format_save" $n_frames "$flow_model" "$output_path"

echo "end $n_frames"

n_frames=15
output_path="$common_out/n_frames_$n_frames/$normalize"
echo "start $normalize $n_frames.."

time bash "$raft_script" "$data_root" $start_index $data_num "$subset" "$format_save" $n_frames "$flow_model" "$output_path" "$normalize"

echo "end $n_frames"

cp "$log_file" "$output"

END_TIMESTAMP=$(date '+%s')
E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
# echo $COPY_E_TIME
bash "$htime_bash" $E_TIME "total exec time"
