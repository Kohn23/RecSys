import math
import random
import torch
import numpy as np


def sequence_crop(seq, length, eta=0.6):
    """From CL4SRec"""
    num_left = math.floor(length * eta)
    crop_begin = random.randint(0, length - num_left)
    croped_item_seq = np.zeros(seq.shape[0])
    if crop_begin + num_left < seq.shape[0]:
        croped_item_seq[:num_left] = seq[crop_begin:crop_begin + num_left]
    else:
        croped_item_seq[:num_left] = seq[crop_begin:]
    return torch.tensor(croped_item_seq, dtype=torch.long), torch.tensor(num_left, dtype=torch.long)


def sequence_mask(seq, length, item_num, gamma=0.3):
    """From CL4SRec"""
    num_mask = math.floor(length * gamma)
    mask_index = random.sample(range(length), k=num_mask)
    masked_item_seq = seq[:]
    masked_item_seq[mask_index] = item_num  # token 0 has been used for semantic masking
    return masked_item_seq, length


def sequence_reorder(seq, length, beta=0.6):
    """From CL4SRec"""
    num_reorder = math.floor(length * beta)
    reorder_begin = random.randint(0, length - num_reorder)
    reordered_item_seq = seq[:]
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
    return reordered_item_seq, length
