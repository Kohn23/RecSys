"""
GAC
################################################

Reference:
    Qi Zhang et al. " Gating augmented capsule network for sequential recommendation."
    in Knowledge-BasedSystems 2022.

Note:

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class DynamicRoutingItem(nn.Module):
    """Item-routing"""

    def __init__(self, config, embedding_size, num_iterations=3):
        super(DynamicRoutingItem, self).__init__()
        self.num_iterations = num_iterations
        self.embedding_size = embedding_size

    def forward(self, capsules, weight_matrices):
        """
        capsules: [batch_size, num_capsules, capsule_dim] = [batch_size, L, embedding_size]
        weight_matrices: [batch_size, num_capsules, capsule_dim, out_capsule_dim] = [batch_size, L, embedding_size, embedding_size]
        """
        batch_size, num_capsules, capsule_dim = capsules.size()
        out_capsule_dim = weight_matrices.size(3)

        b = torch.zeros(batch_size, num_capsules, device=capsules.device)

        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=1)  # [batch_size, num_capsules]

            # capsules: [batch_size, num_capsules, capsule_dim]
            # weight_matrices: [batch_size, num_capsules, capsule_dim, out_capsule_dim]
            # preds: [batch_size, num_capsules, out_capsule_dim]
            preds = torch.matmul(capsules.unsqueeze(2), weight_matrices).squeeze(2)

            s = torch.sum(c.unsqueeze(-1) * preds, dim=1)  # [batch_size, out_capsule_dim]

            v = self.squash(s)

            if iteration < self.num_iterations - 1:
                agreement = torch.sum(preds * v.unsqueeze(1), dim=-1)  # [batch_size, num_capsules]
                b = b + agreement

        return v

    def squash(self, x):
        norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        norm = torch.sqrt(norm_sq)
        return (norm_sq / (1 + norm_sq)) * (x / (norm + 1e-8))


class DynamicRoutingFactor(nn.Module):
    """Factor-routing"""

    def __init__(self, config, embedding_size, num_iterations=3):
        super(DynamicRoutingFactor, self).__init__()
        self.num_iterations = num_iterations
        self.embedding_size = embedding_size

    def forward(self, capsules, weight_matrices):
        """
        capsules: [batch_size, num_capsules, capsule_dim] = [batch_size, embedding_size, L]
        weight_matrices: [batch_size, num_capsules, capsule_dim, out_capsule_dim] = [batch_size, embedding_size, L, embedding_size]
        """
        batch_size, num_capsules, capsule_dim = capsules.size()
        out_capsule_dim = weight_matrices.size(3)

        b = torch.zeros(batch_size, num_capsules, device=capsules.device)

        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=1)  # [batch_size, num_capsules]

            # capsules: [batch_size, num_capsules, capsule_dim]
            # weight_matrices: [batch_size, num_capsules, capsule_dim, out_capsule_dim]
            # preds: [batch_size, num_capsules, out_capsule_dim]
            preds = torch.matmul(capsules.unsqueeze(2), weight_matrices).squeeze(2)

            s = torch.sum(c.unsqueeze(-1) * preds, dim=1)  # [batch_size, out_capsule_dim]

            v = self.squash(s)  # [batch_size, out_capsule_dim]

            if iteration < self.num_iterations - 1:
                agreement = torch.sum(preds * v.unsqueeze(1), dim=-1)  # [batch_size, num_capsules]
                b = b + agreement

        return v

    def squash(self, x):
        norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        norm = torch.sqrt(norm_sq)
        return (norm_sq / (1 + norm_sq)) * (x / (norm + 1e-8))


class GAC(SequentialRecommender):
    r"""
    Gating augmented capsule network for sequential recommendation
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(GAC, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.loss_type = config['loss_type']
        self.num_iterations = config['num_iterations']
        self.L = config['L']
        self.n_users = dataset.num(self.USER_ID)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_embedding_out = nn.Embedding(self.n_items, self.embedding_size)

        self.apply(self._init_weights)

        self.W_g1 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.W_g2 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.W_g3 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))

        # item-routing & factor-routing
        self.W_item = nn.Parameter(torch.randn(self.L, self.embedding_size, self.embedding_size))
        self.W_factor = nn.Parameter(torch.randn(self.embedding_size, self.L, self.embedding_size))

        self.W_g4 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.W_g5 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.W_g6 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.W_g7 = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))

        self.item_routing = DynamicRoutingItem(config, self.embedding_size, self.num_iterations)
        self.factor_routing = DynamicRoutingFactor(config, self.embedding_size, self.num_iterations)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Make sure 'loss_type' in ['BPR', 'CE']! But got {self.loss_type}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def personalized_gating(self, item_embs, user_emb):
        """
        item_embs: [batch_size, L, embedding_size]
        user_emb: [batch_size, embedding_size]
        """
        batch_size, seq_len, emb_size = item_embs.size()
        user_emb_expanded = user_emb.unsqueeze(1).expand(-1, seq_len, -1)

        gate = torch.sigmoid(
            torch.matmul(item_embs, self.W_g1) +
            torch.matmul(user_emb_expanded, self.W_g2) +
            torch.matmul(item_embs * user_emb_expanded, self.W_g3)
        )

        gated_item_embs = item_embs * gate

        return gated_item_embs

    def item_routing_component(self, gated_item_embs):
        batch_size, seq_len, emb_size = gated_item_embs.size()

        capsules = gated_item_embs  # [batch_size, L, embedding_size]

        # self.W_item: [L, embedding_size, embedding_size]
        # weight_matrices: [batch_size, L, embedding_size, embedding_size]
        weight_matrices = self.W_item.unsqueeze(0).expand(batch_size, -1, -1, -1)

        f_caps = self.item_routing(capsules, weight_matrices)

        sum_gated = torch.sum(gated_item_embs, dim=1)

        gate_fusion = torch.sigmoid(
            torch.matmul(sum_gated, self.W_g4) +
            torch.matmul(f_caps, self.W_g5)
        )

        # item-level
        f_item = gate_fusion * f_caps + (1 - gate_fusion) * sum_gated

        return f_item

    def factor_routing_component(self, gated_item_embs):
        batch_size, seq_len, emb_size = gated_item_embs.size()

        capsules = gated_item_embs.permute(0, 2, 1)  # [batch_size, embedding_size, L]

        # self.W_factor: [embedding_size, L, embedding_size]
        # weight_matrices: [batch_size, embedding_size, L, embedding_size]
        weight_matrices = self.W_factor.unsqueeze(0).expand(batch_size, -1, -1, -1)

        f_caps_prime = self.factor_routing(capsules, weight_matrices)

        avg_gated = torch.mean(gated_item_embs, dim=1)

        gate_fusion_prime = torch.sigmoid(
            torch.matmul(avg_gated, self.W_g6) +
            torch.matmul(f_caps_prime, self.W_g7)
        )

        # factor-level
        f_factor = gate_fusion_prime * f_caps_prime + (1 - gate_fusion_prime) * avg_gated

        return f_factor

    def forward(self, interaction):
        user_ids = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        batch_size = user_ids.size(0)

        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_size]
        item_embs = self.item_embedding(item_seq)  # [batch_size, max_seq_len, embedding_size]

        seq_len = item_embs.size(1)
        if seq_len > self.L:
            item_embs = item_embs[:, -self.L:, :]
        elif seq_len < self.L:
            pad_size = self.L - seq_len
            pad_emb = torch.zeros(batch_size, pad_size, self.embedding_size, device=item_embs.device)
            item_embs = torch.cat([item_embs, pad_emb], dim=1)

        gated_item_embs = self.personalized_gating(item_embs, user_emb)

        # item-routing
        f_item = self.item_routing_component(gated_item_embs)

        # factor-routing
        f_factor = self.factor_routing_component(gated_item_embs)

        short_term = f_item + f_factor
        long_term = user_emb

        user_rep = short_term + long_term  # [batch_size, embedding_size] ✅

        return user_rep

    def calculate_loss(self, interaction):
        user_rep = self.forward(interaction)  # [batch_size, embedding_size]
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'CE':
            all_items_emb = self.item_embedding_out.weight  # [n_items, embedding_size]
            all_scores = torch.matmul(user_rep, all_items_emb.transpose(0, 1))  # [batch_size, n_items]

            loss = self.loss_fct(all_scores, pos_items)
            return loss

        else:  # BPR loss
            neg_items = interaction[self.NEG_ITEM_ID]

            pos_item_emb = self.item_embedding_out(pos_items)  # [batch_size, embedding_size]
            neg_item_emb = self.item_embedding_out(neg_items)  # [batch_size, embedding_size]

            pos_scores = torch.sum(user_rep * pos_item_emb, dim=1)  # [batch_size]
            neg_scores = torch.sum(user_rep * neg_item_emb, dim=1)  # [batch_size]

            loss = self.loss_fct(pos_scores, neg_scores)
            return loss

    def predict(self, interaction):
        user_rep = self.forward(interaction)  # [batch_size, embedding_size]
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding_out(test_item)  # [batch_size, embedding_size]

        scores = torch.mul(user_rep, test_item_emb).sum(dim=1)  # [batch_size]
        return scores

    def full_sort_predict(self, interaction):
        user_rep = self.forward(interaction)  # [batch_size, embedding_size]
        all_items_emb = self.item_embedding_out.weight  # [n_items, embedding_size]

        scores = torch.matmul(user_rep, all_items_emb.transpose(0, 1))  # [batch_size, n_items]
        return scores
