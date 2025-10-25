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
