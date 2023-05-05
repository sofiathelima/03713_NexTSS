#!/bin/bash

# Path to our window be file data
# bed_path='/ocean/projects/bio230007p/psimpson/group4/data/CAGE-Seq/FANTOM/hg38_fair+new_CAGE_peaks_phase1and2.bed'
bed_path=$1

# Path to hg38 genome fasta
# genome_path="/ocean/projects/bio230007p/psimpson/group4/data/reference_genomes/hg38.fa"
genome_path=$2

# Path to output file
# out_path="/ocean/projects/bio230007p/psimpson/group4/data/nt_windows/fasta/positive/negative_complement.bed"
out_path=$3

module load bedtools

bedtools complement -i $bed_path -g $genome_path > $out_path

