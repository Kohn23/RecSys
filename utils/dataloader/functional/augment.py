import math
import random
import numpy as np
import torch


def seq_item_crop(seq, length, eta=0.6, device=None):
    """Random crop for item sequence."""
    crop_len = math.floor(length * eta)
    if crop_len == 0:
        return (
            torch.tensor(seq, dtype=torch.long, device=device),
            torch.tensor(length, dtype=torch.long, device=device)
        )

    crop_begin = random.randint(0, length - crop_len)
    crop_item_seq = np.zeros(seq.shape[0])

    if crop_begin + crop_len < seq.shape[0]:
        crop_item_seq[:crop_len] = seq[crop_begin: crop_begin + crop_len]
    else:
        crop_item_seq[:crop_len] = seq[crop_begin:]

    return (
        torch.tensor(crop_item_seq, dtype=torch.long, device=device),
        torch.tensor(crop_len, dtype=torch.long, device=device)
    )


def seq_item_mask(seq, length, gamma=0.3, device=None):
    """Mask operation for item sequence."""
    num_mask = math.floor(length * gamma)
    if num_mask == 0:
        return (
            torch.tensor(seq, dtype=torch.long, device=device),
            torch.tensor(length, dtype=torch.long, device=device)
        )

    mask_index = random.sample(range(length), k=num_mask)
    masked_item_seq = seq.copy()
    masked_item_seq[mask_index] = 0

    return (
        torch.tensor(masked_item_seq, dtype=torch.long, device=device),
        torch.tensor(length, dtype=torch.long, device=device)
    )


def seq_item_noise(seq, length, gamma=0.3, item_num=None, device=None):
    """Noise operation for item sequence with random items."""
    num_noise = math.floor(length * gamma)
    if num_noise == 0:
        return (
            torch.tensor(seq, dtype=torch.long, device=device),
            torch.tensor(length, dtype=torch.long, device=device)
        )

    noise_index = random.sample(range(length), k=num_noise)
    noise_item_seq = seq.copy()
    for index in noise_index:
        noise_item_seq[index] = random.randint(1, item_num)

    return (
        torch.tensor(noise_item_seq, dtype=torch.long, device=device),
        torch.tensor(length, dtype=torch.long, device=device)
    )


def seq_item_reorder(seq, length, beta=0.6, device=None):
    """Reorder operation for item sequence."""
    reorder_len = math.floor(length * beta)
    if reorder_len == 0:
        return (
            torch.tensor(seq, dtype=torch.long, device=device),
            torch.tensor(length, dtype=torch.long, device=device)
        )

    reorder_begin = random.randint(0, length - reorder_len)
    reorder_item_seq = seq.copy()

    shuffle_index = list(range(reorder_begin, reorder_begin + reorder_len))
    random.shuffle(shuffle_index)
    reorder_item_seq[reorder_begin: reorder_begin + reorder_len] = (
        reorder_item_seq[shuffle_index]
    )

    return (
        torch.tensor(reorder_item_seq, dtype=torch.long, device=device),
        torch.tensor(length, dtype=torch.long, device=device)
    )