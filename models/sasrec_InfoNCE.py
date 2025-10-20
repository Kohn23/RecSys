import torch
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec

from models.functional import info_nce


class SASRecInfoNCE(SASRec):
    def __init__(self, config, dataset):
        """
        Note:
            Be aware that 'loss_type' can't be none since this model inherit from a builtin model
            , and we choose neg-sampling within mini-batch
        """
        super().__init__(config, dataset)
        self.tau = config['tau']

    def _sample_neg_items(self, pos_items, batch_size, num_negs=10):
        """
            in-batch neg-sampling
        """
        neg_items = torch.zeros(batch_size, num_negs, dtype=torch.long).to(pos_items.device)

        for i in range(batch_size):
            all_items = torch.arange(1, self.n_items).to(pos_items.device)

            mask = all_items != pos_items[i]
            candidate_neg_items = all_items[mask]

            if len(candidate_neg_items) >= num_negs:
                selected = torch.randperm(len(candidate_neg_items))[:num_negs]
                neg_items[i] = candidate_neg_items[selected]
            else:
                selected = torch.randint(0, len(candidate_neg_items), (num_negs,))
                neg_items[i] = candidate_neg_items[selected]

        return neg_items

    def calculate_loss(self, interaction):
        """
            Calculate InfoNCE loss using contrastive learning from CL4SR
            ，featuring in-batch negatives
        """
        item_seq = interaction[self.ITEM_SEQ]  # [B, T]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]
        pos_items = interaction[self.ITEM_ID]  # [B]
        seq_output = self.forward(item_seq, item_seq_len)  # [B, H]

        batch_size = item_seq.size(0)

        pos_emb = self.item_embedding(pos_items)  # [B, H]

        logits, labels = info_nce(seq_output, pos_emb, self.tau, batch_size, sim='dot')

        return F.cross_entropy(logits, labels)
