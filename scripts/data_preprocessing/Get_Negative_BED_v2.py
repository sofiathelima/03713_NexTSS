import pandas as pd
import numpy as np

import sys

def RandomWindowsFromChromosome(complement_df, chr, n_windows,window_size=600):
    # Choose a region from the chromosome
    chr_regions = complement_df[complement_df.chr == chr]

    # if len(chr_regions) == 0:
    #     return None

    # Randomly pick regions to generate windows from
    sampled_regions = chr_regions.sample(n=n_windows, replace = True).reset_index(drop=True)
    # bed_regions_df = sampled_regions.copy()

    # Randomly Generate windows
    random_starts = []
    for _, row in sampled_regions.iterrows():
        random_starts.append(np.random.randint(row.start,row.end,1).item(0))

    bed_regions_df = pd.DataFrame({"chr":[chr]*len(random_starts),
                                   'start':random_starts,
                                   "end":np.array(random_starts)+window_size}
                                )

    return bed_regions_df


if __name__ == "__main__":
    # Data paths
    # comp_path = '/Users/parkersimpson/CMU/03713/human_negative_complement_copy.bed'
    # pos_windows_path = '/Users/parkersimpson/CMU/03713/human_PositiveSampleWindows.bed'
    # bed_save_path = './test.bed'

    # Get paths from arguments
    comp_path = sys.argv[1]
    pos_windows_path = sys.argv[2]
    bed_save_path = sys.argv[3]
    window_size = 600


    #Load data
    positive_windows_df = pd.read_csv(pos_windows_path,sep='\t',names=['chr','start','end','name'])

    complement_df = pd.read_csv(comp_path,sep='\t',names=['chr','start','end'])
    complement_df.start = complement_df.start.astype(int)
    complement_df.end = complement_df.end.astype(int)

    # Filter complements for only chromosomes in the positive data
    filtered_complement_df = complement_df[(complement_df.end - complement_df.start) >= window_size]
    filtered_complement_df = filtered_complement_df[filtered_complement_df.chr.isin(positive_windows_df.chr.unique().tolist())]

    # Get the count of sample per chromosome in the positive samples
    positive_windows_chr_counts = positive_windows_df.groupby(by=['chr']).size()

    # Generate a balanced amount of negative samples per chormosome
    chr_windows_list = []

    for chr, n_samples in positive_windows_chr_counts.iteritems():
        chr_samples_df = RandomWindowsFromChromosome(filtered_complement_df, chr, n_samples,window_size=window_size)
        # if chr_samples_df is not None:
        chr_windows_list.append(chr_samples_df)

    # Concatenate all negative samples per chromosome
    negative_samples_df = pd.concat(chr_windows_list)

    # Save samples to bed file
    negative_samples_df.to_csv(bed_save_path, sep='\t',header=False,index=False)



