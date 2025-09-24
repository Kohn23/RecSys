import math
import random

import torch
import numpy as np


def item_crop(item_seq, item_seq_len, eta=0.6):
    """ augmentation from CL4SR """
    num_left = math.floor(item_seq_len * eta)
    crop_begin = random.randint(0, item_seq_len - num_left)
    cropped_item_seq = np.zeros(item_seq.shape[0])
    if crop_begin + num_left < item_seq.shape[0]:
        cropped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
    else:
        cropped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:]
    return torch.tensor(cropped_item_seq, dtype=torch.long, device=item_seq.device), \
        torch.tensor(num_left, dtype=torch.long, device=item_seq.device)


def item_mask(item_seq, item_seq_len, n_items, gamma=0.3):
    """ augmentation from CL4SR """
    num_mask = math.floor(item_seq_len * gamma)
    mask_index = random.sample(range(item_seq_len), k=num_mask)
    masked_item_seq = item_seq.cpu().detach().numpy().copy()
    masked_item_seq[mask_index] = n_items  # token 0 has been used for semantic masking
    return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len


def item_reorder(item_seq, item_seq_len, beta=0.6):
    """ augmentation from CL4SR """
    num_reorder = math.floor(item_seq_len * beta)
    reorder_begin = random.randint(0, item_seq_len - num_reorder)
    reordered_item_seq = item_seq.cpu().detach().numpy().copy()
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
    return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len
