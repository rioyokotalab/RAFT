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
img_path="images/$subset"
root_path="$root/$img_path"
local_root=${HINADORI_LOCAL_SCRATCH:-"$script_dir"}
train_path="$local_root/$img_path"

echo ""
echo "start index of videos: " "$2"
echo "number of videos: " "$3"
echo "subset : " "$subset"
echo "root : " "$root_path"
echo "local path: " "$train_path"
echo ""

images_dir=$(find "$root_path" -maxdepth 1 -mindepth 1 -type d | sort | tail -n +"$2" | head -n "$3")
mkdir -p "$train_path"
echo "$images_dir" | parallel cp -R "{}" "$train_path"
