# nohup python pat2onehot.py --folder_path /data2/yixuan/cfDNA/UXM_hg38_refs/selected_ref --output_path /data2/yixuan/cfDNA/UXM_hg38_refs/selected_ref > pat2onehot_new.log 2>&1 &

# nohup python sum_500_CpG_folder_reindex.py --index_bridge_file /data2/yixuan/cfDNA/UXM_hg38_refs/nature_atlas_wgbs_cpg_bed_hg38.bed --region_bin_file /mnt/nas/user/yixuan/cfDNA/CelFEER/data/read_bins.txt --tissue_cpg_folder /data2/yixuan/cfDNA/UXM_hg38_refs/selected_ref --output_path /data2/yixuan/cfDNA/UXM_hg38_refs/sum_500 > sum_500_CpG.log 2>&1 &
# nohup python sum_500_CpG_folder.py --region_bin_file /mnt/nas/user/yixuan/cfDNA/CelFEER/data/read_bins.txt --tissue_cpg_folder /data2/yixuan/cfDNA/UXM_39_refs/new --output_path /data2/yixuan/cfDNA/UXM_39_refs/sum_500 > sum_500_CpG.log 2>&1 &

# nohup python sum_block_CpG_folder.py --region_bin_file /mnt/nas/user/yixuan/cfDNA/UXM_39_refs/processing_scripts/read_top25_block.txt --tissue_cpg_folder /data2/yixuan/cfDNA/UXM_39_refs/selected_ref --output_path /data2/yixuan/cfDNA/UXM_39_refs/sum_block_25 > sum_CpG.log 2>&1 &
# nohup python sum_block_CpG_folder.py --region_bin_file /mnt/nas/user/yixuan/cfDNA/UXM_39_refs/processing_scripts/read_top25_block.txt --tissue_cpg_folder /data2/yixuan/cfDNA/UXM_39_refs/new --output_path /data2/yixuan/cfDNA/UXM_39_refs/sum_block_25 > sum_CpG.log 2>&1 &

# nohup python sum_block_CpG_folder.py --region_bin_file /mnt/nas/user/yixuan/cfDNA/UXM_39_refs/processing_scripts/read_top25_block.txt --tissue_cpg_folder /mnt/nas/user/yixuan/cfDNA/UXM_39_refs/cfDNA_source_data --output_path /mnt/nas/user/yixuan/cfDNA/UXM_39_refs/sum_block_25_cfDNA > sum_CpG.log 2>&1 &

# nohup python combine_all_cell_types.py --sum_folder /data2/yixuan/cfDNA/UXM_hg38_refs/sum_500 --output_path /data2/yixuan/cfDNA//UXM_hg38_refs/UXM_hg38_ref.bed > combine.log 2>&1 &
# nohup python combine_all_cell_types.py --sum_folder /data2/yixuan/cfDNA/UXM_hg38_refs/sum_500 --output_path /data2/yixuan/cfDNA/UXM_hg38_refs/UXM_hg38_ref_5.bed > combine.log 2>&1 &

# python combine_all_cell_types.py --sum_folder /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/sum_500 --output_path /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/UXM_hg38_ref_25.bed
nohup python combine_all_cell_types.py --sum_folder /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/sum_500 --output_path /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/cfSort_WBC_major/UXM_hg38_ref_WBC.bed  > combine.log 2>&1 &

# nohup python combine_all_cell_types.py --sum_folder /data2/yixuan/cfDNA/UXM_39_refs/sum_block_25 --output_path /data2/yixuan/cfDNA/UXM_39_refs/UXM39_block_markered_25.bed > combine.log 2>&1 &

# python ref_sum_folder.py --region_bin_file read_top25_block.txt --tissue_cpg_folder /data2/yixuan/cfDNA/UXM_hg38_refs/sum_500 --output_path /data2/yixuan/cfDNA/UXM_hg38_refs/sum_500

# python pat2onehot_temp.py --folder_path /data2/yixuan/cfDNA/UXM_hg38_refs/selected_ref --output_path /data2/yixuan/cfDNA/UXM_hg38_refs/selected_ref > debug.log 2>&1 &
# python ref_sum_folder.py --region_bin_file /mnt/nas/user/yixuan/cfDNA/CelFEER/data/read_bins.txt --tissue_cpg_folder /data2/yixuan/cfDNA/UXM_39_refs/sum_500 --output_path /data2/yixuan/cfDNA/UXM_39_refs/sum_500