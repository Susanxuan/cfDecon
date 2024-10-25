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
    # file of CpGs
    with open(file, "r") as input_file:
        cpg_file = csv.reader(input_file, delimiter="\t")
        next(cpg_file, None)
        # get methylation read counts for each position
        temp_key_index=0
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
                else:
                    print(chrom, start)
                    print(chr, num1, num2)   
                    print(i, "failed")                
                    assert 1<0,'bad comparison'
            
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

    tissue=os.listdir(tissue_cpg_folder)
    print(tissue)


    for folder in tissue:
        folder_dir=os.path.join(tissue_cpg_folder, folder)
        if os.path.isdir(folder_dir):
            samples= os.listdir(folder_dir)
            sample_values=[]
            output_file = os.path.join(output_path, folder.split('/')[-1]+ '.bed')
            for sample in samples:
                if sample.endswith('.bed'):
                    tissue_cpg_file = os.path.join(folder_dir, sample)
                    print(folder.split('/')[-1])
                    temp=pd.read_csv(tissue_cpg_file, header=None, sep='\t')
                    # print(temp.iloc[:,3:].values.shape)
                    # print(temp.iloc[:,3:].values)
                    sample_values.append(temp.iloc[:,3:].values)
                    
            mean_array = np.mean(sample_values, axis=0)
            # print(mean_array.shape)
            # print(mean_array)
            temp.iloc[:,3:]=mean_array
            print(temp)
            temp.to_csv(output_file,header=False,index=False,sep='\t')
