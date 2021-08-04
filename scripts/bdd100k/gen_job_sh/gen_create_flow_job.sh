#!/bin/bash
readonly program=$(basename $0)
readonly args=(bash_name out_job_dir bdd100k_root log_dir file_num datanum subset nodename nodenum exectime)

function print_usage_and_exit() {
    echo >&2 "Usage: ${program} $(IFS=' '; echo ${args[*]^^})"
    exit 1
}

if [ $# -gt ${#args[@]} ] || [ $# -lt 7 ]; then
    print_usage_and_exit
fi

for arg in ${args[@]}; do
    eval "readonly ${arg}=$1"
    shift
done

git_root=$(git rev-parse --show-toplevel | head -1)
is_abci=$(uname -a | grep "abci")
is_abci_a=$(echo "$is_abci" | grep "\-a")

script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
script_abs_dir_path=$(
    cd "$(dirname "$bash_name")" || exit
    pwd
)
script_name="$(basename "$bash_name")"
out_dir=$(
    cd "$out_job_dir" || exit
    pwd
)
data_root=$(
    cd "$bdd100k_root" || exit
    pwd
)
result_dir_comm="$(dirname "$log_dir")/$(basename "$log_dir")"
if [ -d "$result_dir_comm" ]; then
    result_dir_comm=$(
        cd "$log_dir" || exit
        pwd
    )
else
    if [ ${result_dir_comm:0:1} == "." ]; then
        result_dir_comm=${result_dir_comm##*"../"}
    fi
    if [ ${result_dir_comm:0:1} != "/" ] && [ ${result_dir_comm:0:1} != "~" ]; then
        result_dir_comm="$git_root/$result_dir_comm"
    fi
fi
if [ -z "$exectime" ]; then
    time="12:00:00"
else
    time="$exectime"
fi
node_name="$nodename"
if [ -z "$node_name" ]; then
    node_name="am_1"
fi

pushd "$out_dir"


for i in $(seq $file_num);
do
    echo "$i"
    number=$(($i-1))
    folder_number="$(printf %03d $i)"
    file_number="$(printf %03d $number)"
    file_name=""$subset"_flow_job_"$file_number".sh"
    if [ -f "$file_name" ]; then
        rm "$file_name"
    fi
    echo "$file_name"
    result_dir="$result_dir_comm/$datanum/$subset/$folder_number"
    job_name=""$subset"_"$folder_number"_gen"
    if type "qsub" > /dev/null 2>&1
    then
        if [ "$is_abci" ]; then
            if [ "$is_abci_a" ];then
                node_name="rt_AG.small"
            else
                node_name="rt_G.small"
            fi
        else
            node_name="s_gpu"
        fi
    fi
    bash "$script_dir/gen_job_header.sh" "$file_name" "$node_name" "$nodenum" "$time" "$job_name" "$result_dir"
    echo "source /etc/profile.d/modules.sh" >> "$file_name"
    echo "" >> "$file_name"
    if type "qsub" > /dev/null 2>&1
    then
        if [ "$is_abci" ]; then
            echo "module load cuda/11.1/11.1.1" >> "$file_name"
        else
            echo "module load cuda/11.0.194" >> "$file_name"
        fi
    elif type "ybatch" > /dev/null 2>&1
    then
        echo "module load cuda/11.1"
    fi
    echo "" >> "$file_name"
    echo "bdd100k_root_path=\""$data_root"\"" >> "$file_name"
    echo "bdd100k_flow_script_path=\"\$bdd100k_root_path/flow/scripts\"" >> "$file_name"
    echo "scripts_path=\""$script_abs_dir_path"\"" >> "$file_name"
    echo "data_num="$datanum"" >> "$file_name"
    echo "start_num=\$(("$number" * \$data_num))" >> "$file_name"
    echo "echo \"data num : \$data_num\"" >> "$file_name"
    echo "echo \"start : \$start_num\"" >> "$file_name"
    echo "" >> "$file_name"
    echo "mkdir -p \"\$bdd100k_flow_script_path\"" >> "$file_name"
    echo "" >> "$file_name"
    echo "time bash \"\$scripts_path/"$script_name"\" \"\$bdd100k_root_path\" \$start_num \$data_num \""$subset"\"" >> "$file_name"
    echo "" >> "$file_name"
    echo "cp \"\$scripts_path/"$script_name"\" \"\$bdd100k_flow_script_path\"" >> "$file_name"
    echo "cp "$file_name" \"\$bdd100k_flow_script_path\"" >> "$file_name"
done

popd
