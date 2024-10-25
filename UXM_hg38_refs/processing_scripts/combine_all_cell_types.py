import csv
import numpy as np
import pandas as pd
import sys
import math
import collections
import argparse
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Process a folder and output one-hot encoded data.")
    parser.add_argument("--sum_folder", type=str, required=True, help="Path to the input file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    sum_folder = args.sum_folder # the input file
    output_path = args.output_path
 
    for r in range(1,10):
        bed_files = [file for file in sorted(os.listdir(sum_folder)) if file.endswith('.bed')]
        dfs = []
        ct =[]
        Blood_files = []
        random_files = []
        Blood_files = [file for file in bed_files if file.startswith('Blood')] #WBC major
        random_f = random.sample(list(set(bed_files) - set(Blood_files)), r)
        # print(Blood_files)
        # print(random_f)
        random_files = Blood_files + random_f

        # random_files = random.sample(bed_files, 25) #5 10 15 20 25

        for filename in random_files:
            if filename.endswith('.bed'):
                tissue_cpg_file = os.path.join(sum_folder, filename)
                ct.append(filename.split('.')[0])
                print(filename.split('.')[0])
                cpg_file_temp = pd.read_csv(tissue_cpg_file, delimiter="\t", header=None)
                # print(cpg_file_temp.head())
                last_five_cols = cpg_file_temp.iloc[:, -5:]  # Select the last five columns
                # print(sum(last_five_cols[3].values))
                dfs.append(last_five_cols)
        print('num cell types', len(random_files))
        result = pd.concat(dfs, axis=1)
        # print(result.shape)
        ref_1=pd.concat([cpg_file_temp.iloc[:, :3],result], axis=1)
        # print(ref_1)
        ref_1 = ref_1.rename(columns={ref_1.columns[0]: 'chrom', ref_1.columns[1]: 'start', ref_1.columns[2]: 'end'})

        ref_1.to_csv(output_path[:-4]+'_'+str(r)+'.bed',sep='\t',index=False)
        # Open the file in write mode
        with open(output_path[:-4]+'_'+str(r)+'_ct.txt', 'w') as file:
            # Iterate over the list
            for item in ct:
                # Write each item to a new line in the file
                file.write(item + '\n')

