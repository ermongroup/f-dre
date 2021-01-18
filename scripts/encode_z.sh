#!/bin/bash

PERCS=(0.1 0.25 0.5 1.0)

for perc in "${PERCS[@]}"
do
    echo "running flow encoding for perc=${perc}"
    CUDA_VISIBLE_DEVICES=1 python maf.py --data_dir='/atlas/u/kechoi/multi-fairgen/data' --output_dir="/atlas/u/kechoi/multi-fairgen/results/subset_maf_perc${perc}" --model='maf' --dataset='MNISTSubset_combined_z' --hidden_size=1024 --restore_file="/atlas/u/madeline/multi-fairgen/results/maf/subset_perc${perc}/best_model_checkpoint.pt" --encode --subset=True --perc=${perc}
done