#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Tongyue Sun
Date: 2023-07-21

Description:
    Converting PAT files into One-hot encoded format tailored for the CelFEER.
"""
import pandas as pd
import numpy as np
import os
import csv
import gc
import argparse

# gzip -dk *.gz
# nohup python pat2onehot.py > pat2onehot.log 2>&1 &
def process_line(row):
    reads = row['Reads'].replace('.', '')  # 去除 reads 中的 '.'
    if len(reads) < 3:
        return None
    c_ratio = reads.count('C') / len(reads)
    oh_idx = None
    
    if 0.125 <= c_ratio < 0.375:
        oh_idx = 1
    elif 0.375 <= c_ratio < 0.625:
        oh_idx = 2
    elif 0.625 <= c_ratio < 0.875:
        oh_idx = 3
    elif 0.875 <= c_ratio <= 1:
        oh_idx = 4
    else:
        oh_idx = 0
    
    one_hot = np.zeros(5)
    one_hot[oh_idx] = row['Reads_number']
    # print(row['Reads_number'] )
    return one_hot.astype(int)

def process_file(file_path, output_file):
    header_written = False
    for df in pd.read_csv(file_path, sep='\t', chunksize=100000, 
                      names=['chr', 'CpG_idx', 'Reads', 'Reads_number', 'Origin']):
        df['OneHot'] = df.apply(process_line, axis=1)
        df = df.dropna(subset=['OneHot'])
        # # 将OneHot转换为字符串,用制表符\t分隔
        df['OneHot'] = df['OneHot'].apply(lambda x: '\t'.join(map(str, x)))  # 输出为TSV文件
        if not header_written:
            df.to_csv(output_file, sep='\t', index=False, columns=['chr','CpG_idx','OneHot'])
            header_written = True
        else:  
            df.to_csv(output_file, mode='a', header=False, sep='\t', index=False, columns=['chr','CpG_idx','OneHot'])
        del df

    with open(output_file) as f:
        # 删除pandas额外添加内容
        content = ''.join(line.replace('"', '') for line in f)

    with open(output_file, 'w') as f:
        f.write(content)    

    
def process_folder(folder_path, output_path):    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.pat'):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            output_file = os.path.join(output_path, filename.split('.')[0] + '_onehot_records.tsv')
            if filename.split('.')[0] + '_onehot_records.tsv' in os.listdir(output_path):
                print('skip ',filename.split('.')[0] + '_onehot_records.tsv')
            else:
                print('start ', filename.split('.')[0] + '_onehot_records.tsv')
            
                process_file(file_path, output_file)
                gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="Process a folder and output one-hot encoded data.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the input folder.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    folder_path = args.folder_path
    output_path = args.output_path
    process_folder(folder_path, output_path)

if __name__ == "__main__":
    main()
