#!/bin/bash

readonly program=$(basename $0)
readonly args=(job_dir start_line job_num subset)

function print_usage_and_exit() {
    echo >&2 "Usage: ${program} $(IFS=' '; echo ${args[*]^^})"
    exit 1
}

if [ $# -ne ${#args[@]} ]; then
    print_usage_and_exit
fi

for arg in ${args[@]}; do
    eval "readonly ${arg}=$1"
    shift
done

job_root=$(
    cd "$job_dir" || exit
    pwd
)
start_job_index=$(($start_line - 1))
end_job_index=$(($job_num + $start_line - 2))
is_abci=$(uname -a | grep "abci")

pushd "$job_root"

job_list=$(find "$job_root" -name ""$subset"_flow_job_*.sh" | sort | tail -n +"$start_line" | head -n "$job_num")

echo "$start_job_index"-"$end_job_index"
echo "$job_list"
for i in ${job_list};
do
    file_name=$(basename "$i")
    if type "qsub" > /dev/null 2>&1
    then
        if [ "$is_abci" ]; then
            echo "qsub -g gcd50666 "$file_name""
            qsub -g gcd50666 "$file_name"
        else
            echo "qsub -g tga-RLA "$file_name""
            qsub -g tga-RLA "$file_name"
        fi
    elif type "ybatch" > /dev/null 2>&1
    then
        echo "ybatch "$file_name""
        ybatch "$file_name"
    fi
done

popd
