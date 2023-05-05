#!/bin/bash

# Path to our CAGEseq data
# cage_path='/ocean/projects/bio230007p/psimpson/group4/data/CAGE-Seq/FANTOM/hg38_fair+new_CAGE_peaks_phase1and2.bed'
cage_path=$1

# Path to our TFBS data
# epd_path='/ocean/projects/bio230007p/psimpson/group4/data/CAGE-Seq/EPD/human_epdnew_Pdv2W.bed'
epd_path=$2

# Path to output file
# out_path="/ocean/projects/bio230007p/psimpson/group4/data/intersect_fantom-hg38_epd.bed"
out_path=$3

module load bedtools

bedtools intersect -a $cage_path -b $epd_path > $out_path

