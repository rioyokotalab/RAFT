#!/bin/bash

readonly program=$(basename $0)
readonly args=(bash_name nodename nodenum exectime jobname outdir is_err)

function print_usage_and_exit() {
    echo >&2 "Usage: ${program} $(IFS=' '; echo ${args[*]^^})"
    exit 1
}

if [ $# -gt ${#args[@]} ] || [ $# -lt 1 ]; then
    print_usage_and_exit
fi

for arg in ${args[@]}; do
    eval "readonly ${arg}=$1"
    shift
done

git_root=$(git rev-parse --show-toplevel | head -1)
is_abci=$(uname -a | grep "abci")
is_abci=$(echo "$is_abci" | grep "\-a")

file_dir_name=$(
    cd "$(dirname "$bash_name")" || exit
    pwd
)
file_name="$file_dir_name/$(basename "$bash_name")"

node_name="$nodename"
if [ -z "$node_name" ]; then
    if type "qsub" > /dev/null 2>&1
    then
        node_name="s_core"
        if [ "$is_abci" ]; then
           node_name="rt_C.small"
           if [ "$is_abci_a" ]; then
              node_name="rt_AG.small"
           fi
        fi
    elif type "ybatch" > /dev/null 2>&1
    then
        node_name="any_1"
    fi
fi
node_num="$nodenum"
if [ -z "$node_num" ]; then
    node_num="1"
fi
time="$exectime"
if [ -z "$time" ] && [ -z "$is_abci" ]; then
    time="24:00:00"
fi
job_name="$jobname"
if [ -z "$job_name" ]; then
    job_name="test_job"
fi
out_dir="$outdir"
if [ -n "$out_dir" ]; then
    if [ -d "$out_dir" ]; then
        out_dir=$(
            cd "$6" || exit
            pwd
        )
    else
        if [ ${out_dir:0:1} == "." ]; then
            out_dir=${out_dir##*"../"}
        fi
        if [ ${out_dir:0:1} != "/" ] && [ ${out_dir:0:1} != "~" ]; then
            out_dir="$git_root/$out_dir"
        fi
    fi
fi

echo "#!/bin/bash" > "$file_name"
if type "qsub" > /dev/null 2>&1
then
    echo "#$ -cwd" >> "$file_name"
    echo "#$ -l "$node_name"="$node_num"" >> "$file_name"
    if [ -n "$time" ]; then
        echo "#$ -l h_rt="$time"" >> "$file_name"
    fi
    echo "#$ -N "$job_name"" >> "$file_name"
    if [ -n "$out_dir" ]; then
        echo "#$ -o "$out_dir"/"$job_name"_\$JOB_ID.out" >> "$file_name"
    fi
    if [ -n "$is_err" ]; then
        echo "#$ -e "$out_dir"/"$job_name"_\$JOB_ID.err" >> "$file_name"
    else
        if [ -n "$out_dir" ]; then
            echo "#$ -j y" >> "$file_name"
        fi
    fi
elif type "ybatch" > /dev/null 2>&1
then
    echo "#YBATCH -r "$node_name"" >> "$file_name"
    echo "#SBATCH -N "$node_num"" >> "$file_name"
    echo "#SBATCH -J "$job_name"" >> "$file_name"
    if [ -n "$time" ]; then
        echo "#SBATCH --time="$time"" >> "$file_name"
    fi
    if [ -n "$out_dir" ]; then
        echo "#SBATCH --output "$out_dir"/"$job_name"_slurm%j.out" >> "$file_name"
    fi
    if [ -n "$is_err" ]; then
        echo "#SBATCH --error "$out_dir"/"$job_name"_slurm%j.err" >> "$file_name"
    fi
fi
if [ -n "$out_dir" ]; then
    echo "" >> "$file_name"
    echo "mkdir -p \""$out_dir"\"" >> "$file_name"
fi
echo "" >> "$file_name"
