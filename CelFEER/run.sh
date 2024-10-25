python scripts/celfeer_WGBS_sim.py data/WGBS_sim_input.txt output/WGBS_sim 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_500.pkl

python scripts/celfeer_WGBS_sim.py data/WGBS_sim_input.txt output/WGBS_sim/1000_7_spar0.3 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_1000_7_spar0.3.pkl 

python scripts/celfeer_WGBS_sim.py data/WGBS_sim_input.txt output/WGBS_sim/1000_7_spar0 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_1000_7_spar0.pkl 

# python scripts/AE_WGBS_sim.py data/WGBS_sim_input.txt output/AE_sim 7 

# python scripts/AE_WGBS_sim.py data/WGBS_sim_input.txt output/AE_sim 7 \
#     -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_500.pkl

python scripts/Multi_channel_AE_WGBS_sim.py data/WGBS_sim_input.txt output/Multi_channel_AE_sim 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_500.pkl

# python scripts/multi_channel_AE.py data/WGBS_sim_input.txt output/multi_channel_AE_sim 7 \
#     -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_500.pkl

# python scripts/celfeer.py data/ALS.txt output/ALS 8 

# python scripts/Multi_channel_AE_celfeer.py data/ALS.txt output/Multi_channel_ALS 8 \
#     -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/Multi_channel_ALS/sample_array_1000_19.pkl

python scripts/Multi_channel_AE_celfeer.py data/WGBS_sim_input.txt output/Multi_channel_WGBS/two_sim_spar0.3 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_1000_7_spar0.3.pkl


python scripts/Multi_channel_AE_celfeer.py data/WGBS_sim_input.txt output/Multi_channel_WGBS/two_sim_spar0 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_1000_7_spar0.pkl

# one signature prediction
python scripts/Multi_channel_AE_celfeer.py data/WGBS_sim_input.txt output/Multi_channel_WGBS/one_sim_spar0.3 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_1000_7_spar0.3.pkl
    -s 1

python scripts/Multi_channel_AE_celfeer.py data/WGBS_sim_input.txt output/Multi_channel_WGBS/one_sim_spar0 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_1000_7_spar0.pkl \
    -s 1

python scripts/Multi_channel_AE_celfeer.py data/WGBS_sim_input.txt output/Multi_channel_WGBS/one_sim_spar0.3 7 \
    -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_1000_7_spar0.3.pkl \
    -s 1 

    

# python scripts/celfeer_WGBS_sim.py data/WGBS_sim_input.txt output/WGBS_sim 7 \
#     -f /mnt/nas/user/yixuan/cfDNA/CelFEER/output/WGBS_sim/sample_array_500.pkl

# python scripts/celfeer_generated_sim.py output/generated_sim 15 7 700 500 500 
# python scripts/Multi_channel_AE_generated_sim.py output/Multi_channel_AE_generated_sim 15 7 700 500 500 


#######new commands#######

markers=121
markers=211
markers=122
markers=212

CUDA_VISIBLE_DEVICES=0 python scripts/Multi_channel_AE_celfeer.py data/WGBS_ref1_inter${markers}.txt new_data_output/Multi_channel_WGBS/one_sim_spar0.3_${markers} 10 \
    -f new_data_output/fractions/sample_array_1000_10_spar0.3.pkl \
    -s 1 

CUDA_VISIBLE_DEVICES=0 python scripts/Multi_channel_AE_celfeer.py data/WGBS_ref1_inter${markers}.txt new_data_output/Multi_channel_WGBS/one_sim_spar0.3_${markers} 10 \
    -f new_data_output/fractions/sample_array_1000_10_spar0.3.pkl \
    -s 1 \
    -a

CUDA_VISIBLE_DEVICES=1 python scripts/Multi_channel_AE_celfeer.py data/WGBS_ref1_inter${markers}.txt new_data_output/Multi_channel_WGBS/one_sim_spar0.3_${markers}_high 10 \
    -f new_data_output/fractions/sample_array_1000_10_spar0.3.pkl \
    -s 1 \
    -a \
    -m high-resolution 

python scripts/celfeer_WGBS_sim.py data/WGBS_ref1_inter121.txt new_data_output/WGBS_sim/1000_10_spar0.3_121 10 \
    -f new_data_output/fractions/sample_array_1000_10_spar0.3.pkl 


python scripts/markers.py data/WGBS_ref1.bed data/WGBS_ref1_markers.txt 1000 10 20 0 True original