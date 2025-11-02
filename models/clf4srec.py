# -*- coding: utf-8 -*-

"""
CLF4SRec
################################################

Reference:
     Yichi Zhang et al. "Contrastive Learning with Frequency Domain for Sequential Recommendation."
     in Applied Soft Computing 2023.

Reference:


"""

import torch
from torch import nn
from recbole.model.sequential_recommender import SASRec
from models.layers import BandedFourierLayer
from models.functional import info_nce, seq_fft


class CLF4SRec(SASRec):
    r"""

    """
    def __init__(self, config, dataset):
        super(CLF4SRec, self).__init__(config, dataset)

        self.lmd = config['lmd']
        self.lmd_tf = config['lmd_tf']
        self.tau = config['tau']
        self.sim = config['sim']
        self.beta = config['beta']

        self.fft_layer = BandedFourierLayer(self.hidden_size, self.hidden_size, 0, 1, length=self.max_seq_length)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization again
        self.apply(self._init_weights)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output_t = trm_output[-1]
        output_t = self.gather_indexes(output_t, item_seq_len - 1)

        output_f = self.fft_layer(input_emb)
        trm_output_f = self.trm_encoder(output_f, extended_attention_mask, output_all_encoded_layers=True)
        output_f = trm_output_f[-1]
        output_f = self.gather_indexes(output_f, item_seq_len - 1)

        return output_t, output_f  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output_t, seq_output_f = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output_t * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output_t * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'

            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(seq_output_t, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # NCE
        # Time ssl
        aug_item_seq, aug_len = interaction['aug1'], interaction['aug_len1']
        aug_seq_output_t, aug_seq_output_f = self.forward(aug_item_seq, aug_len)
        nce_logits_t, nce_labels_t = info_nce(aug_seq_output_t, seq_output_t, self.tau, seq_output_t.shape[0], self.sim)
        nce_loss_t = self.nce_fct(nce_logits_t, nce_labels_t)

        # Time-Frequency ssl
        nce_logits_t_f, nce_labels_t_f = info_nce(seq_output_f, seq_output_t, self.tau, seq_output_t.shape[0], self.sim)
        nce_loss_t_f = self.nce_fct(nce_logits_t_f, nce_labels_t_f)

        # Frequency ssl
        f_aug_seq_output_amp, f_aug_seq_output_phase = seq_fft(seq_output_t)
        f_seq_output_amp, f_seq_output_phase = seq_fft(seq_output_f)

        # Amp ssl
        nce_logits_amp, nce_labels_amp = info_nce(f_aug_seq_output_amp, f_seq_output_amp, self.tau, seq_output_t.shape[0], self.sim)
        nce_loss_amp = self.nce_fct(nce_logits_amp, nce_labels_amp)

        # Phase ssl
        nce_logits_phase, nce_labels_phase = info_nce(f_aug_seq_output_phase, f_seq_output_phase, self.tau, seq_output_t.shape[0], self.sim)
        nce_loss_phase = self.nce_fct(nce_logits_phase, nce_labels_phase)

        return loss + self.lmd/2 * (
                    self.lmd_tf * nce_loss_t + (1 - self.lmd_tf)/3 * (nce_loss_t_f + nce_loss_phase + nce_loss_amp))

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
