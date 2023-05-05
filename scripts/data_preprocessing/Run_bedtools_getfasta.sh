#!/bin/bash


# Path to our window be file data
# bed_path='/ocean/projects/bio230007p/psimpson/group4/data/nt_windows/bed/positive/PositiveSampleWindows.bed'
bed_path=$1

# Path to hg38 genome fasta
# genome_path="/ocean/projects/bio230007p/psimpson/group4/data/reference_genomes/hg38.fa"
genome_path=$2

# Path to output file
# out_path="/ocean/projects/bio230007p/psimpson/group4/data/nt_windows/fasta/positive/positive_windows.fa"
out_path=$3

module load bedtools

bedtools getfasta -fi $genome_path -bed $bed_path -fo $out_path


