#!/bin/bash
readonly program=$(basename $0)
readonly args=(bash_name bdd100k_root node_name node_num time)

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

script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
bash_file_dirname=$(
    cd "$(dirname "$bash_name")" || exit
    pwd
)
bash_filename="$bash_file_dirname/$(basename "$bash_name")"
job_name="$(basename "$bash_name" ".sh")"

git_root=$(git rev-parse --show-toplevel | head -1)

data_root="$bdd100k_root"
if [ -n "$data_root" ]; then
    data_root=$(
        cd "$data_root" || exit
        pwd
    )
fi

out_dir="$git_root/bash_output"
is_err=""

job_dir_name="$git_root/jobs/bash_job"
mkdir -p "$job_dir_name"

job_filename="$job_dir_name/$(basename "$bash_filename")"

bash "$script_dir/gen_job_header.sh" "$job_filename" "$node_name" "$node_num" "$time" "$job_name" "$out_dir" "$is_err"
if [ -n "$data_root" ]; then
    echo "time bash \""$bash_filename"\" \""$data_root"\"" >> "$job_filename"
else
    tail -n +2 "$bash_filename" >> "$job_filename"
fi
