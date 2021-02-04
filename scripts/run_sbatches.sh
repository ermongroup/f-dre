#!/bin/bash

for experiment in /atlas/u/madeline/multi-fairgen/scripts/feb2_dre_x_fair_generate_same_bkgd/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch --partition=atlas --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas7,atlas8,atlas9,atlas10,atlas11,atlas12,atlas13,atlas14,atlas15 --time=7-00:00:00 --cpus-per-task=8 --gres=gpu:1 --mem=32G --nodes=1 --ntasks-per-node=1 --error=${experiment}_err.log --output=${experiment}_out.log $experiment
    sleep 1
done

# done
echo "Done"
