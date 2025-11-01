# -*- coding: utf-8 -*-

"""
CL4SRec
################################################

Reference:
    Wang-Cheng Kang et al. "Contrastive Learning for Sequential Recommendation." in ICDE 2022.

Reference:
    https://github.com/JamZheng/CL4SRec-pytorch

"""
import torch
from torch import nn

from recbole.model.sequential_recommender import SASRec
from models.functional import info_nce
import torch.nn.functional as F


class CL4SRec(SASRec):
    r"""
    Note:
        This is currently a test model
    """

    def __init__(self, config, dataset):
        super(CL4SRec, self).__init__(config, dataset)

        # params for Info_nce
        self.lmd = config['lmd']
        self.tau = config['tau']
        self.sim = config['sim']

    def _contrastive_loss(self, interaction):
        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
            interaction['aug1'], interaction['aug_len1'], interaction['aug2'], interaction['aug_len2']
        seq_output1 = self.forward(aug_item_seq1, aug_len1)
        seq_output2 = self.forward(aug_item_seq2, aug_len2)

        nce_logits, nce_labels = info_nce(seq_output1, seq_output2, self.tau, aug_len1.shape[0], self.sim)

        return F.cross_entropy(nce_logits, nce_labels)

    def calculate_loss(self, interaction):
        loss = super(CL4SRec, self).calculate_loss(interaction)
        nce_loss = self._contrastive_loss(interaction)
        return loss + self.lmd * nce_loss


