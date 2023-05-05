"""
This file will generate the .bed file that is used to obtain nuclotide sequences from the regions that define our data
"""
import sys

# Path to bed file with data
# bed_path = '/ocean/projects/bio230007p/psimpson/group4/data/intersect_fantom-hg38_epd.bed'
bed_path = sys.argv[1]

# Save path
# save_path = "/ocean/projects/bio230007p/psimpson/group4/data/nt_windows/bed/positive/PositiveSampleWindows.bed"
save_path = sys.argv[2]

new_bed_lines = []
with open(bed_path) as f:
    # Map hit IDs to sequence regions

    for line in f:
        # Split line by tabs
        line=line.split('\t')

        # Define the distance from the start of the peak reagion to its center
        diff=int((int(line[2])-int(line[1]))/2)

        # Define coordinate for 600 nt window centered at the peaks
        window_start, window_end = int(line[1])+diff-300, int(line[1])+diff+300

        if window_start > 0:
            new_bed_lines.append(f"{line[0]}\t{window_start}\t{window_end}\t{line[3]}\n")


    #print(coords) 

with open(save_path,'w') as f:
    # Write new bed file with the liens we made
    f.writelines(new_bed_lines)
