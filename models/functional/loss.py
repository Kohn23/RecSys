import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_correlated_samples(batch_size):
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


def info_nce(query_emb, pos_emb, temperature, batch_size, sim='dot'):
    """
    In-batch InfoNCE loss (SimCLR-style)

    Args:
        query_emb: [B, H]   anchor / user sequence representation
        pos_emb:   [B, H]   positive item representation
        temperature: float  scaling factor
        batch_size: int
        sim: 'dot' or 'cos'
    Returns:
        logits: [2B, 1 + 2(B-1)]   positive + negatives
        labels: [2B]               index of positive sample (always 0)
    """
    N = 2 * batch_size  # total pairs

    if query_emb.size(1) != pos_emb.size(1):
        query_emb = query_emb.expand(-1, pos_emb.size(1))
    all_emb = torch.cat((query_emb, pos_emb), dim=0)  # [2B, H]

    if sim == 'cos':
        sim_matrix = nn.functional.cosine_similarity(
            all_emb.unsqueeze(1), all_emb.unsqueeze(0), dim=2
        ) / temperature
    elif sim == 'dot':
        sim_matrix = torch.mm(all_emb, all_emb.T) / temperature
    else:
        raise NotImplementedError

    sim_q_to_p = torch.diag(sim_matrix, batch_size)   # [B]
    sim_p_to_q = torch.diag(sim_matrix, -batch_size)  # [B]
    positive_scores = torch.cat((sim_q_to_p, sim_p_to_q), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)
    negative_scores = sim_matrix[mask].reshape(N, -1)

    logits = torch.cat((positive_scores, negative_scores), dim=1)  # [N, 1 + ...]
    labels = torch.zeros(N, dtype=torch.long, device=logits.device)  # 正样本都在 index 0

    return logits, labels
