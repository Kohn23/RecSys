import torch
import torch.nn as nn


def _mask_correlated_samples(batch_size):
    """
    From recbole_da
    """
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def sim_fn(z1: torch.Tensor, z2: torch.Tensor, tau=1.0, fn='cos'):
    if z1.size(1) != z2.size(1):
        z1 = z1.expand(-1, z2.size(1))
    z = torch.cat((z1, z2), dim=0)  # [2B, H]
    if fn == 'cos':
        return nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / tau
    elif fn == 'dot':
        return torch.mm(z, z.T) / tau
    else:
        raise NotImplementedError


def info_nce(query_emb, pos_emb, tau, batch_size, sim='dot'):
    """
    In-batch InfoNCE loss (SimCLR-style)

    Args:
        query_emb: [B, H]   anchor / user sequence representation
        pos_emb:   [B, H]   positive item representation
        tau: float  scaling factor
        batch_size: int
        sim_fn: 'dot' or 'cos'
    Returns:
        logits: [2B, 1 + 2(B-1)]   positive + negatives
        labels: [2B]               index of positive sample (always 0)
    """
    N = 2 * batch_size  # total pairs

    sim_matrix = sim_fn(query_emb, pos_emb, tau, sim)

    sim_q_to_p = torch.diag(sim_matrix, batch_size)   # [B]
    sim_p_to_q = torch.diag(sim_matrix, -batch_size)  # [B]
    positive_scores = torch.cat((sim_q_to_p, sim_p_to_q), dim=0).reshape(N, 1)

    mask = _mask_correlated_samples(batch_size)
    negative_scores = sim_matrix[mask].reshape(N, -1)

    logits = torch.cat((positive_scores, negative_scores), dim=1)  # [N, 1 + ...]
    labels = torch.zeros(N, dtype=torch.long, device=logits.device)  # 正样本都在 index 0

    return logits, labels
