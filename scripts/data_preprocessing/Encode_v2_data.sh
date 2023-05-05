#!/bin/bash

# data verison
data_version=$1
# echo $data_version

# Species of the data
species=$2
# echo $species

# Path to nt_window directory
nt_window_path=$3
# echo $nt_window

# Make directory to store data splits
data_dir="processed_data/model_data_${data_version}/${species}"
# echo $data_dir
mkdir -p $data_dir

# Encode positive samples
seq_fasta="${nt_window_path}/fasta/positive/${species}_positive_windows_NoN.fa"
# echo $seq_fasta
python scripts/data_preprocessing/feature_generation/encode_fasta.py $seq_fasta $data_dir 1
# eval "python /ocean/projects/bio230007p/psimpson/group4/scripts/data_preprocessing/feature_generation/encode_fasta.py $( echo $seq_fasta ) $( echo $data_dir ) 1"

# Encode Negative Samples
seq_fasta="${nt_window_path}/fasta/negative/${species}_negative_windows_NoN.fa"
# echo $seq_fasta
python scripts/data_preprocessing/feature_generation/encode_fasta.py $seq_fasta $data_dir 0
# eval "python /ocean/projects/bio230007p/psimpson/group4/scripts/data_preprocessing/feature_generation/encode_fasta.py $( echo $seq_fasta ) $( echo $data_dir ) 0"

