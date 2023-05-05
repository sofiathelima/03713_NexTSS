#!/bin/bash

input_f1=$1
output_f1=$2

# Replace every occurence of N with either A,C,G,or T at random
awk -v target_char="N" -v predef_list="ACTG" 'BEGIN {OFS=FS=""} {for (i=1; i<=NF; i++) if (toupper($i) == target_char) $i = substr(predef_list, int(rand() * length(predef_list)) + 1, 1)} 1' $input_f1 > $output_f1
