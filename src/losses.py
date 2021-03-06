import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def joint_gen_disc_loss(model, x, logits, y, alpha):
    """
    joint generative/discriminative approach
    """
    clf_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())
    flow_loss = -model.flow.log_prob(x).mean(0)
    # TODO: double check shapes
    loss = (alpha * clf_loss) + ((1.-alpha) * flow_loss)

    return loss


def grad_penalty(model, alpha, device, norm_type='fro'):
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type) for (n,p) in model.named_parameters() if 'fc' in n
        ]), norm_type
    )
    return (alpha * total_norm)
