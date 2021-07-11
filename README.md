# Featurized Density Ratio Estimation
This repo contains a reference implementation for featurized density ratio estimation (f-dre) as described in the paper:
> Featurized Density Ratio Estimation </br>
> [Kristy Choi*](http://kristychoi.com/), [Madeline Liao*](https://www.linkedin.com/in/madelineliao/), [Stefano Ermon](https://cs.stanford.edu/~ermon/) </br>
> Uncertainty in Artificial Intelligence (UAI), 2020. </br>
> Paper: https://arxiv.org/abs/2107.02212 </br>


## For non-KMM/KLIEP experiments:
### 1) Environment setup:
(a) Necessary packages can be found in `src/requirements.txt`.

(b) Set the correct Python path using the following command:
```
source init_env.sh
```
Once this step is completed, all further steps for running experiments should be launched from the `src/` directory.

### 2) Pre-train the normalizing flow (Masked Autoregressive Flow). 
As the experimental workflow is quite similar across datasets, we'll use the toy Gaussian mixtures as a concrete example. To first train the flow, run:
```
python3 main.py --config flows/gmm/maf.yaml --exp_id=gmm_flow --ni
```
Config files for other datasets can be found in `src/configs/flows/<dataset>/`. The trained flow will be saved in `src/flows/results/<exp_id>` (note the path where it is saved) and will be used for downstream z-space density ratio estimation.


### 3) Generate encodings prior to running binary classification.
The following script will use the pre-trained flow in Step #2 to generate encodings of the data points:
```
python3 main.py --config flows/gmm/maf.yaml --exp_id encode_gmm_z --restore_file=./flows/results/gmm_flow/ --sample --encode_z --ni
```

### 4) Train density ratio estimator (classifier) on the encoded data points.
Running the following script will estimate density ratios in feature space:
```
python3 main.py --classify --config classification/gmm/mlp_z.yaml  --exp_id gmm_z --ni
```
Note that config files for other baselines such as training on the x's directly and joint training of the flow and classifier can be found in `classification/gmm/<name_of_method>.yaml`. The scripts to run these methods are the same as the above, just with a modification of the `exp_id`.

### 5) (For MNIST targeted generation experiment only) Generate samples!
#### 5.1) Train an attribute classifier
In order to get stats on the generated samples, train an attribute classifer using the following script:
```
python3 main.py  \
          --classify  \
          --attr background  \ # or "digits"
          --config classification/mnist/diff_bkgd/attr_bkgd.yaml   \
          --exp_id classify_attr_bkgd  \
          --ni
```
#### 5.2) Generate the samples
The following script performs regular, unweighted generation. To perform reweighted generation with DRE in z-space, replace `--generate_samples` with `--fair_generate`. To perform this with DRE in x-space, replace `--generate_samples` with `--fair_generate` and `--dre_x`.
```
python3 main.py  \
         --sample  \
         --seed 10 \
         --generate_samples \
         --config flows/mnist/diff_bkgd/perc0.1.yaml   \
         --exp_id regular_generation_perc0.1  \
         --attr_clf_ckpt=/classification/results/{path to attr classifier checkpoint.pth} \
         --restore_file=/flows/results/{directory with flow checkpoint} \
         --ni
```
## For KMM/KLIEP experiments:
Each of these experiments is self-contained within one Jupyter notebook. Simply run the corresponding notebook cell by cell in `/notebooks`.

## References
If you find this work useful in your research, please consider citing the following paper:
```
@article{choi2021featurized,
  title={Featurized Density Ratio Estimation},
  author={Choi, Kristy and Liao, Madeline and Ermon, Stefano},
  journal={arXiv preprint arXiv:2107.02212},
  year={2021}
}
```
