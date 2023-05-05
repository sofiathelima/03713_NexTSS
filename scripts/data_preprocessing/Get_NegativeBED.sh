#!/bin/bash

py_path=$1
complement_path=$2
positives_path=$3
out_path=$4

# module load anaconda3
# conda activate dl

python $py_path $complement_path $positives_path $out_path
