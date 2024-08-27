import pandas as pd

# 定义列名，因为 BED 文件通常没有表
sample_file_path = 'UXM_hg38_ref_31_100m.txt'
sample_data = pd.read_csv(sample_file_path, header=None, delimiter="\t")

sample_data = sample_data.iloc[:, :8]

reference_file = 'UXM_hg38_ref_31_100m.txt'
reference_file = pd.read_csv(reference_file, header=None, delimiter="\t")

# 拼接数据
combined_data = pd.concat([sample_data, reference_file], axis=1)

# 保存拼接后的数据到新的文件
combined_file_path = 'UXM_ref31_100m.bed'
combined_data.to_csv(combined_file_path, header=False, index=False, sep="\t", float_format='%.6f')

