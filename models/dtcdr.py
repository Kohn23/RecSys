import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.loss import BPRLoss


class DTCDR(CrossDomainRecommender):
    r"""
    A minimal implementation of DTCDR (Deep Transfer Cross-Domain Recommendation).

    Paper: Z. Hu et al., "Cross-Domain Recommendation: Challenges, Progress and Prospects,"
    in IJCAI 2018 (original DTCDR idea).

    This is a **simplified version** that only demonstrates pipeline compatibility.
    """

    def __init__(self, config, dataset):
        super(DTCDR, self).__init__(config, dataset)

        # basic parameters
        self.embedding_size = config["embedding_size"]

        # source & target domain user/item embeddings
        self.user_embedding_source = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding_source = nn.Embedding(self.n_items, self.embedding_size)

        self.user_embedding_target = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding_target = nn.Embedding(self.n_items, self.embedding_size)

        # projection layers for knowledge transfer
        self.transfer_layer = nn.Linear(self.embedding_size, self.embedding_size)

        # loss
        self.loss_type = config["loss_type"]
        self.bpr_loss = BPRLoss()

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, user, item, domain="target"):
        if domain == "source":
            user_e = self.user_embedding_source(user)
            item_e = self.item_embedding_source(item)
        else:
            user_e = self.user_embedding_target(user)
            item_e = self.item_embedding_target(item)

        # transfer knowledge from source to target
        transferred_user = self.transfer_layer(user_e)
        return (transferred_user * item_e).sum(dim=-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]

        # target domain prediction
        pos_score = self.forward(user, pos_item, domain="target")

        # negative sampling
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_score = self.forward(user, neg_item, domain="target")

        loss = self.bpr_loss(pos_score, neg_score)

        # + alignment loss between source and target embeddings
        user_source = self.user_embedding_source(user)
        user_target = self.user_embedding_target(user)
        align_loss = F.mse_loss(self.transfer_layer(user_source), user_target)

        return loss + 0.1 * align_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # 默认 target domain
        score = self.forward(user, item, domain="target")
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding_target(user)
        all_items = self.item_embedding_target.weight
        scores = torch.matmul(user_e, all_items.t())
        return scores.view(-1)
