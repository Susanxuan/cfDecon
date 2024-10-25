# imports
import csv
import numpy as np
import pandas as pd
import sys
import math
import collections
import argparse
import os

def get_methylation_counts(file):
    """
    add together the methylation values for all CpGs in the selected region
    """

    # file of CpGs
    with open(file, "r") as input_file:
        cpg_file = csv.reader(input_file, delimiter="\t")
        next(cpg_file, None)
        # get methylation read counts for each position
        for line in cpg_file:
            chrom, start = line[0], int(line[1])
            meth = np.array(line[2:], dtype=np.float64)
            binstart = 500*math.floor(start/500)
            if (chrom, binstart) in regions_dict:
                regions_dict[(chrom, binstart)] += meth

    return regions_dict

def write_region_file(output_file, regions_dict):
    """
    write region file
    """
    with open(output_file, "w") as output:
        bed_file = csv.writer(output, delimiter="\t",  lineterminator="\n")

        for chrom, start in regions_dict:
            values = regions_dict[(chrom, start)]
            bed_file.writerow(
                [chrom] + [start] + [start+499] + list(values)
            )
