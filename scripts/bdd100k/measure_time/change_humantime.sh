#!/bin/bash

target_time=$1
minutes=$(($target_time / 60))
# echo "$minutes m"
seconds=$(($target_time - ($minutes * 60)))
# echo "$seconds s"
hours=$(($minutes / 60))
# echo "$hours h"
minutes=$(($minutes - $hours * 60))
# echo "$minutes m"

echo "$2: $hours h $minutes m $seconds s"
