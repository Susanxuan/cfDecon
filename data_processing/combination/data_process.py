import pandas as pd

# 定义列名，因为 BED 文件通常没有表
sample_file_path = 'GSM6810003_CNVS-NORM-110000263-cfDNA-WGBS-Rep1_sum_block.bed'
dtype = {0: str, 1: int, 2: int, 3: float, 4: float, 5: float, 6: float, 7: float}  # 假设列 0 是字符串，1 和 2 是整数，3 到 7 是浮点数
sample_data = pd.read_csv(sample_file_path, header=None, delimiter="\t", dtype=dtype)

# 增加 comment 行
comment_line = pd.DataFrame([["chrom", "start", "end", "3", "4", "5", "6", "7"]])
sample_df = pd.concat([comment_line, sample_data], ignore_index=True)

float_columns = [3, 4, 5, 6, 7]  # 指定浮点数列
for col in float_columns:
    sample_df[col] = sample_df[col].apply(lambda x: '{:.6f}'.format(float(x)) if pd.notnull(x) else x)


reference_file = 'UXM_39ref_1000m.txt'
reference_file = pd.read_csv(reference_file, header=None, delimiter="\t")

# 拼接数据
combined_data = pd.concat([sample_df, reference_file], axis=1)

# 保存拼接后的数据到新的文件
combined_file_path = 'UXM_39ref_1000m.bed'
combined_data.to_csv(combined_file_path, header=False, index=False, sep="\t", float_format='%.6f')

