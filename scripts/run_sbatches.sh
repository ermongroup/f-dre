#!/bin/bash

for experiment in /atlas/u/kechoi/multi-fairgen/scripts/slurm/joint_gmm_flow_mlp_perc1.0_alpha0.00*
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
    sleep 1
done

# done
echo "Done"
