#!/bin/bash
script_dir=$(
    cd "$(dirname "$0")" || exit
    pwd
)
root=$(
    cd "$1" || exit
    pwd
)

subset="$4"
local_root=${HINADORI_LOCAL_SCRATCH:-"$script_dir"}

echo ""
echo "start index of zips: " "$2"
echo "number of zips: " "$3"
echo "subset : " "$subset"
echo "root : " "$root"
echo "local path: " "$local_root"
echo ""

zips=$(find "$root" -maxdepth 1 -type f -name "bdd100k_videos_$subset_*.zip" | sort | tail -n +"$2" | head -n "$3")
echo "$zips" | parallel -u unzip -q {} -d "$local_root"
# echo "$zips"
