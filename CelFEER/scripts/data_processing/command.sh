# python /data2/yixuan/cfDNA/CelFEER/scripts/data_processing/bismark_meth_to_input.py /data2/yixuan/cfDNA/CelFEER/scripts/data_processing/temp /data2/yixuan/cfDNA/data_by_yumei/adipose_rep2/ENCFF312IXF.filter.sort.peread.txt 
# python /data2/yixuan/cfDNA/CelFEER/scripts/data_processing/bismark_meth_to_input_revised.py /data2/yixuan/cfDNA/CelFEER/scripts/data_processing/temp /data2/yixuan/cfDNA/data_by_yumei/adipose_rep2/ENCFF312IXF.filter.sort.peread.txt 

# cell_type=adipose_rep1
# sample_id=ENCFF990GGC

# cell_type=adrenal_rep1
# sample_id=ENCFF568VSU

# cell_type=adrenal_rep2
# sample_id=ENCFF255BVZ

# cell_type=heart_rep1
# sample_id=ENCFF118LHX

# cell_type=heart_rep2
# sample_id=ENCFF903GQY

# cell_type=intestine_rep1
# sample_id=ENCFF491SJO

# cell_type=intestine_rep2
# sample_id=ENCFF766QRU

# cell_type=lung_rep1
# sample_id=ENCFF342ZGW

# cell_type=lung_rep2
# sample_id=ENCFF198YBR

# cell_type=pancreas_rep1
# sample_id=ENCFF258HZR

# cell_type=pancreas_rep2
# sample_id=ENCFF020QAX

# cell_type=psoas_rep1
# sample_id=ENCFF072RZW

# cell_type=psoas_rep2
# sample_id=ENCFF657BFC

# cell_type=sigmoid_rep1
# sample_id=ENCFF033UPY

# cell_type=sigmoid_rep2
# sample_id=ENCFF050YKS

# cell_type=spleen_rep1
# sample_id=ENCFF718FGN

# cell_type=spleen_rep2
# sample_id=ENCFF942HFP


processed_bam=/data2/yixuan/cfDNA/data_by_yumei/${cell_type}/${sample_id}.filter.sort.peread.txt 

python /data2/yixuan/cfDNA/CelFEER/scripts/data_processing/bismark_meth_to_input_revised.py /data2/yixuan/cfDNA/CelFEER/scripts/data_processing/temp ${processed_bam}

input_bed=/data2/yixuan/cfDNA/CelFEER/scripts/data_processing/temp/${sample_id}.bed
sum_bed=/data2/yixuan/cfDNA/CelFEER/scripts/data_processing/temp/${sample_id}_sum500.bed
python /data2/yixuan/cfDNA/CelFEER/scripts/data_processing/sum_reads_in_500_bins.py /data2/yixuan/cfDNA/CelFEER/data/read_bins.txt ${input_bed} ${sum_bed}