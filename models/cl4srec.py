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
        # currently using da-style names and fixed augment
        self.aug1 = config["CROP_ITEM_SEQ"]
        self.aug_len1 = config["CROP_ITEM_SEQ_LEN"]
        self.aug2 = config["REORDER_ITEM_SEQ"]
        self.aug_len2 = config["ITEM_SEQ_LEN"]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # NCE
        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
            interaction[self.aug1], interaction[self.aug_len1], interaction[self.aug2], interaction[self.aug_len2]
        seq_output1 = self.forward(aug_item_seq1, aug_len1)
        seq_output2 = self.forward(aug_item_seq2, aug_len2)

        nce_logits, nce_labels = info_nce(seq_output1, seq_output2, self.tau, aug_len1.shape[0], self.sim)

        nce_loss = F.cross_entropy(nce_logits, nce_labels)

        return loss + self.lmd * nce_loss
