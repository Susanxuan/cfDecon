# python scripts/markers.py data/WGBS_ref1.bed data/WGBS_ref1_1000markers.bed 1000 10 20 0 True original
# python scripts/markers.py data/WGBS_ref2.bed data/WGBS_ref2_1000markers.bed 1000 10 20 0 True original


python scripts/markers.py data/WGBS_ref1.bed data/WGBS_ref1_500markers.bed 500 10 20 0 True original
python scripts/markers.py data/WGBS_ref2.bed data/WGBS_ref2_500markers.bed 500 10 20 0 True original

python scripts/markers.py data/WGBS_ref1.bed data/WGBS_ref1_500markers.bed 500 9 20 0 True original
python scripts/markers.py data/WGBS_ref2.bed data/WGBS_ref2_500markers.bed 500 9 20 0 True original

python scripts/markers.py scripts/data_processing/temp/ref_2.bed scripts/data_processing/temp/WGBS_ref2_500markers.bed 500 10 20 0 True original
python scripts/markers.py scripts/data_processing/temp/ref_2.bed scripts/data_processing/temp/WGBS_ref2_100markers.bed 100 10 20 0 True original
python scripts/markers.py scripts/data_processing/temp/ref_1.bed scripts/data_processing/temp/WGBS_ref1_100markers.bed 100 10 20 0 True original




