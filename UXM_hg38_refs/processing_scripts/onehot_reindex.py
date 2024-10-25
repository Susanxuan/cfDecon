#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Tongyue Sun
Date: 2023-07-21

Description:
    Converting CpG index using nature_atals_wgbs_cpg_bed_hg39.bed.
"""
# imports
import csv
import numpy as np
import pandas as pd
import sys
import math
import collections
import argparse
import os

def get_index_dict(file):
    """
    retrieve the 500 bp bins from text file
    """

    regions_dict = collections.defaultdict(list)  

    with open(file, "r") as input_file:
        regions_file = csv.reader(input_file, delimiter="\t")
        next(regions_file, None) 
        for line in regions_file:
            chrom, new, old = line[0], int(line[1]), int(line[2])
            regions_dict[(chrom, old)] = new
            
    return regions_dict


def get_methylation_counts(file, regions_dict):
    """
    add together the methylation values for all CpGs in the selected region
    """

    # file of CpGs
    with open(file, "r") as input_file:
        cpg_file = csv.reader(input_file, delimiter="\t")
        next(cpg_file, None)
        # get methylation read counts for each position
        for line in cpg_file:
            chrom, old = line[0], int(line[1])
            meth = np.array(line[2:], dtype=np.float64)
            if (chrom, binstart) in regions_dict:
                regions_dict[(chrom, binstart)] += meth

    return regions_dict


def write_bed_file(output_file, regions_dict):
    """
    write bed file of summed counts for all tissues
    """
    with open(output_file, "w") as output:
        bed_file = csv.writer(output, delimiter="\t",  lineterminator="\n")

        for chrom, start in regions_dict:
            values = regions_dict[(chrom, start)]
            bed_file.writerow(
                [chrom] + [start] + [start+499] + list(values)
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Process a folder and output one-hot encoded data.")
    parser.add_argument("--index_bradge_file", type=str, required=True, help="Path to the region bins file.")
    parser.add_argument("--tissue_cpg_folder", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    index_bradge_file = args.index_bradge_file
    tissue_cpg_folder = args.tissue_cpg_folder # the input file
    output_path = args.output_path

    regions = get_index_dict(
        index_bradge_file
    )  # get dictionary of regions to sum within

    for filename in os.listdir(tissue_cpg_folder):
        if filename.endswith('_onehot_records.tsv'):
            tissue_cpg_file = os.path.join(tissue_cpg_folder, filename)
            output_file = os.path.join(output_path, filename.split('.')[0][:-15] + '_sum_500.bed')
            if filename.split('.')[0][:-15] + '_sum_500.bed' in os.listdir(output_path):
                # assert 1<0,filename.split('.')[0][:-15]
                print('skip',filename.split('.')[0][:-15])
            else:
                print('start ', filename.split('.')[0][:-15])
                get_methylation_counts(
                    tissue_cpg_file, regions
                )  # get methylation read counts of Cpgs within region
                write_bed_file(output_file, regions)  # write output
                print('done with ', filename.split('.')[0][:-15])