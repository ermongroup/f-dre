#!/bin/bash

# PERCS=(0.1 0.25 0.5 1.0)
# IDXS=(0 1 2)
IDXS=(0)

for i in "${IDXS[@]}"
do
    echo "running flow encoding for idx=${i}"
    CUDA_VISIBLE_DEVICES=0 python3 main.py --trainer=Classifier --config="mnist_subset/resnet_${i}.yaml"
done