import torch
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class SASRecInfoNCE(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.temp = config.get('infonce_temp', 1.0)

    def calculate_loss(self, interaction):
        """
            calculate InfoNCE
        """
        item = interaction[self.ITEM_ID]                # [B, T] or [B]
        neg_key = f'neg_{self.ITEM_ID}'
        if neg_key not in interaction:
            neg_key = 'neg_item_id'
        neg_item = interaction[neg_key]                 # [B, T, K] or [B, K]

        seq_output = self.forward(interaction)          # [B, T, H]
        h = seq_output

        e_pos = self.item_embedding(item)               # [B, T, H]
        if neg_item.dim() == 2 and item.dim() == 2:
            neg_item = neg_item.unsqueeze(1).expand(item.size(0), item.size(1), neg_item.size(-1))

        e_neg = self.item_embedding(neg_item)           # [B, T, K, H]

        pos_logits = (h * e_pos).sum(-1, keepdim=True)  # [B, T, 1]
        neg_logits = (h.unsqueeze(2) * e_neg).sum(-1)   # [B, T, K]

        logits = torch.cat([pos_logits, neg_logits], dim=-1) / self.temp  # [B, T, 1+K]

        item_seq = interaction[self.ITEM_SEQ]           # [B, T]
        valid_mask = (item > 0).float()                 # [B, T]

        log_probs = F.log_softmax(logits, dim=-1)       # [B, T, 1+K]
        loss = -(log_probs[..., 0]) * valid_mask        # [B, T]
        loss = loss.sum() / (valid_mask.sum() + 1e-12)

        return loss
