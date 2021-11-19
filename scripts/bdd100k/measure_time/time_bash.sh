#!/bin/sh
#$ ...
#$ ...
START_TIMESTAMP=$(date '+%s')

# iroiro

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo $E_TIME
