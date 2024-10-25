# imports
import csv
import numpy as np
import pandas as pd
import sys
import math
import collections
import argparse
import os

def get_region_dict(file):
    """
    retrieve the 500 bp bins from text file
    """

    regions_dict = collections.defaultdict(list) 

    with open(file, "r") as input_file:
        regions_file = csv.reader(input_file, delimiter="\t")
        next(regions_file, None) 
        for line in regions_file:
            chrom, start, end = line[0], int(line[1]), int(line[2])
            regions_dict[(chrom, start, end)] = np.zeros(5)
            
    return regions_dict


def get_methylation_counts(file, regions_dict):
    """
    add together the methylation values for all CpGs in the selected region
    """

    regions_dict_keys=list(regions_dict.keys())
    # print(regions_dict_keys)
    # file of CpGs
    with open(file, "r") as input_file:
        cpg_file = csv.reader(input_file, delimiter="\t")
        next(cpg_file, None)
        # get methylation read counts for each position
        temp_key_index=0
        # for line in cpg_file:
        for i, line in enumerate(cpg_file):
            chrom, start = line[0], int(line[1])
            meth = np.array(line[2:], dtype=np.float64)
            if temp_key_index >= len(regions_dict_keys):
                break
            if start < regions_dict_keys[temp_key_index][1]:
                continue
            elif start > regions_dict_keys[temp_key_index][2]:
                temp_key_index+=1
            else:
                chr, num1, num2 = regions_dict_keys[temp_key_index]
                if num1 <= start <= num2:
                    regions_dict[(chrom, num1, num2)] += meth
                    # print(i, "success")
                else:
                    print(chrom, start)
                    print(chr, num1, num2)   
                    print(i, "failed")                
                    assert 1<0,'bad comparison'
                # if int(enumerate(cpg_file)[i+1][1]) > regions_dict_keys[temp_key_index][2]:
                #     temp_key_index+=1

            
    return regions_dict


def write_bed_file(output_file, regions_dict):
    """
    write bed file of summed counts for all tissues
    """
    with open(output_file, "w") as output:
        bed_file = csv.writer(output, delimiter="\t",  lineterminator="\n")

        for chrom, start, end in regions_dict:
            values = regions_dict[(chrom, start, end)]
            bed_file.writerow(
                [chrom] + [start] + [end] + list(values)
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Process a folder and output one-hot encoded data.")
    parser.add_argument("--region_bin_file", type=str, required=True, help="Path to the region bins file.")
    parser.add_argument("--tissue_cpg_folder", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    regions_file = args.region_bin_file
    tissue_cpg_folder = args.tissue_cpg_folder # the input file
    output_path = args.output_path

    print('reading region file')

    regions = get_region_dict(
        regions_file
    )  # get dictionary of regions to sum within

    for filename in os.listdir(tissue_cpg_folder):
        if filename.endswith('_onehot_records.tsv'):
            tissue_cpg_file = os.path.join(tissue_cpg_folder, filename)
            output_file = os.path.join(output_path, filename.split('.')[0][:-15] + '_sum_block.bed')
            # assert 1<0,filename.split('.')[0][:-15]
            print('start ', filename.split('.')[0][:-15])
            get_methylation_counts(
                tissue_cpg_file, regions
            )  # get methylation read counts of Cpgs within region
            write_bed_file(output_file, regions)  # write output
            print('done with ', filename.split('.')[0][:-15])
