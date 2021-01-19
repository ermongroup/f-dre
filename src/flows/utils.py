import torch
import numpy as np


def classify_examples(model, config):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    preds = []
    samples = np.load(config['sample_path'])['x']
    n_batches = samples.shape[0] // 1000

    with torch.no_grad():
        # generate 10K samples
        for i in range(n_batches):
            x = samples[i*1000:(i+1)*1000]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1)
            preds.append(pred)
        preds = torch.cat(preds).data.cpu().numpy()

    return preds


def fairness_discrepancy(data, n_classes):
    """
    computes fairness discrepancy metric for single or multi-attribute
    this metric computes L2, L1, AND KL-total variation distance
    """
    unique, freq = np.unique(data, return_counts=True)
    props = freq / len(data)
    truth = 1./n_classes

    # L2 and L1
    l2_fair_d = np.sqrt(((props - truth)**2).sum())
    l1_fair_d = abs(props - truth).sum()

    # q = props, p = truth
    kl_fair_d = (props * (np.log(props) - np.log(truth))).sum()
    
    return l2_fair_d, l1_fair_d, kl_fair_d