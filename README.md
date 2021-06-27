# Featurized Density Ratio Estimation
This repo contains a reference implementation for featurized density ratio estimation (f-dre) as described in the paper:
> Featurized Density Ratio Estimation </br>
> [Kristy Choi*](http://kristychoi.com/), [Madeline Liao*](https://www.linkedin.com/in/madelineliao/), [Stefano Ermon](https://cs.stanford.edu/~ermon/) </br>
> Uncertainty in Artificial Intelligence (UAI), 2020. </br>
> Paper: https://arxiv.org/abs/TODO </br>


## 1) Environment setup:
(a) Necessary packages can be found in `requirements.txt`. [TODO!]

(b) Set the correct Python path using the following command:
```
source init_env.sh
```
Once this step is completed, all further steps for running experiments should be launched from the `src/` directory.

## 2) Pre-train the normalizing flow (Masked Autoregressive Flow). 
As the experimental workflow is quite similar across datasets, we'll use the toy Gaussian mixtures as a concrete example. To first train the flow, run:
```
python3 main.py --config flows/gmm/maf.yaml --exp_id=gmm_flow --ni
```
Config files for other datasets can be found in `src/configs/flows/<dataset>/`. The trained flow will be saved in `src/flows/results/<exp_id>` (note the path where it is saved) and will be used for downstream z-space density ratio estimation.


## 3) Generate encodings prior to running binary classification.
The following script will use the pre-trained flow in Step #2 to generate encodings of the data points:
```
python3 main.py --config flows/gmm/maf.yaml --exp_id encode_gmm_z --restore_file=./flows/results/gmm_flow/ --sample --encode_z --ni
```


## 4) Train density ratio estimator (classifier) on the encoded data points.
Running the following script will estimate density ratios in feature space:
```
python3 main.py --classify --config classification/gmm/mlp_z.yaml  --exp_id gmm_z --ni
```
Note that config files for other baselines such as training on the x's directly and joint training of the flow and classifier can be found in `classification/gmm/<name_of_method>.yaml`. The scripts to run these methods are the same as the above, just with a modification of the `exp_id`.


## References
If you find this work useful in your research, please consider citing the following paper:
```
TODO
```
