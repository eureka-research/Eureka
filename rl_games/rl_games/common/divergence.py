import torch
import torch.distributions as dist



def d_kl_discrete(p, q):
    # p = target, q = online
    # categorical distribution parametrized by logits
    logits_diff = p - q
    p_probs = torch.exp(p)
    d_kl = (p_probs * logits_diff).sum(-1)
    return d_kl


def d_kl_discrete_list(p, q):
    d_kl = 0
    for pi, qi in zip(p,q):
        d_kl += d_kl_discrete(pi, qi)
    return d_kl

def d_kl_normal(p, q):
    # p = target, q = online
    p_mean, p_sigma = p
    q_mean, q_sigma = q
    mean_diff = ((q_mean - p_mean) / q_sigma).pow(2)
    var_ratio = (p_sigma / q_sigma).pow(2)

    d_kl = 0.5 * (var_ratio + mean_diff - 1 - var_ratio.log())
    return d_kl.sum(-1)