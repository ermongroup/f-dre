import numpy as np
import itertools
import glob
import os


SBATCH_PREFACE = \
"""#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --exclude=atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas7,atlas8,atlas9,atlas10,atlas11,atlas12,atlas13,atlas14,atlas15
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
"""

# constants for commands
# alphas = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# alphas = [0.01, 0.02, 0.05, 0.07]
alphas = [0.001, 0.005, 0.007]

OUTPUT_PATH="/atlas/u/kechoi/multi-fairgen/scripts/slurm/"

# write to file
for alpha in alphas:
    exp_id = 'joint_gmm_flow_mlp_perc1.0_alpha{}'.format(alpha)
    script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
    base_cmd = 'python3 /atlas/u/kechoi/multi-fairgen/src/main.py --classify --config /atlas/u/kechoi/multi-fairgen/src/configs/classification/gmm/joint_sweep/joint_flow_mlp_perc1.0_alpha{}.yaml  --exp_id joint_gmm_flow_mlp_perc1.0_alpha{} --ni'.format(alpha, alpha)
    with open(script_fn, 'w') as f:
        print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
        print(base_cmd, file=f)
        print('echo "running shapes"', file=f)
        print('sleep 1', file=f)