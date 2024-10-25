# nohup python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py /data2/yixuan/cfDNA/UXM_hg38_refs/UXM_hg38_ref.bed /data2/yixuan/cfDNA/UXM_hg38_refs/WGBS_ref_100markers.bed 100 31 20 0 True original > makers_100.log 2>&1 &

# nohup python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py /data2/yixuan/cfDNA/UXM_hg38_refs/UXM_hg38_ref.bed /data2/yixuan/cfDNA/UXM_hg38_refs/WGBS_ref_1000markers.bed 1000 31 20 0 True original > makers_1000.log 2>&1 &

# python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py /data2/yixuan/cfDNA/UXM_hg38_refs/UXM_hg38_ref_5.bed /data2/yixuan/cfDNA/UXM_hg38_refs/WGBS_ref_5_50markers.bed 50 5 20 0 True original
# python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py /data2/yixuan/cfDNA/UXM_hg38_refs/UXM_hg38_ref_10.bed /data2/yixuan/cfDNA/UXM_hg38_refs/WGBS_ref_10_500markers.bed 500 10 20 0 True original
# python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/UXM_hg38_ref_15.bed /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/WGBS_ref_15_100markers.bed 100 15 20 0 True original
# python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/UXM_hg38_ref_20.bed /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/WGBS_ref_20_100markers.bed 100 20 20 0 True original
# python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/UXM_hg38_ref_25.bed /data2/yixuan/cfDNA_Decon/UXM_hg38_refs/WGBS_ref_25_100markers.bed 100 25 20 0 True original

for i in 1 2 3 4 5 6 7 8 9
do
  echo "Running exp with $i, total cells $((i+5))"
  data_name='/data2/yixuan/cfDNA_Decon/UXM_hg38_refs/cfSort_WBC_major/UXM_hg38_ref_WBC_'$i'.bed '
  output_name='/data2/yixuan/cfDNA_Decon/UXM_hg38_refs/cfSort_WBC_major/UXM_hg38_ref_WBC_'$i'_100markers.bed'
  python /mnt/nas/user/yixuan/cfDNA/CelFEER/scripts/markers.py $data_name $output_name 100 $((i+5)) 20 0 True original
  echo "=================="
done

