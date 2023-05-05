#!/bin/bash

# data verison
data_version=$1

# Species of the data
species=$2

# Path to nt_window directory
nt_window_path=$3

py_path=$4

# Make directory to store data splits
data_dir="./processed_data/model_data_${data_version}/${species}"
# data_dir="/ocean/projects/bio230007p/psimpson/group4/data/data_${data_version}/${species}"
# echo $data_dir
mkdir -p $data_dir

# Encode positive samples
seq_fasta="${nt_window_path}/fasta/positive/${species}_positive_windows_NoN.fa"
# echo $seq_fasta
python $py_path $seq_fasta $data_dir 1
# python /ocean/projects/bio230007p/psimpson/group4/scripts/data_preprocessing/feature_generation/encode_fasta.py $seq_fasta $data_dir 1

# Encode Negative Samples
seq_fasta="${nt_window_path}/fasta/negative/${species}_negative_windows_NoN.fa"
# echo $seq_fasta
python $py_path $seq_fasta $data_dir 0

