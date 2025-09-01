from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoNet(CrossDomainRecommender):
    r"""
    CoNet: Collaborative Cross Networks for Cross-Domain Recommendation
    (Hu et al., KDD 2018)
    """
    input_type = 'pair'

    def __init__(self, config, dataset):
        super(CoNet, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        mlp_hidden_size_list = config['mlp_hidden_size_list']

        # Embedding
        self.source_user_embedding = nn.Embedding(self.source_n_users, self.embedding_size)
        self.source_item_embedding = nn.Embedding(self.source_n_items, self.embedding_size)
        self.target_user_embedding = nn.Embedding(self.target_n_users, self.embedding_size)
        self.target_item_embedding = nn.Embedding(self.target_n_items, self.embedding_size)

        # 双域 MLP
        prev_dim = self.embedding_size * 2
        self.source_layers, self.target_layers, self.cross_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for hidden_dim in mlp_hidden_size_list:
            self.source_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.target_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.cross_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            prev_dim = hidden_dim

        self.source_out = nn.Linear(prev_dim, 1)
        self.target_out = nn.Linear(prev_dim, 1)

    def forward(self, user, item, domain="target"):
        if domain == "source":
            u_e = self.source_user_embedding(user)
            i_e = self.source_item_embedding(item)
        else:
            u_e = self.target_user_embedding(user)
            i_e = self.target_item_embedding(item)

        h = torch.cat([u_e, i_e], dim=-1)
        h_s, h_t = h.clone(), h.clone()

        for src_layer, tgt_layer, cross in zip(self.source_layers, self.target_layers, self.cross_layers):
            h_s_ = F.relu(src_layer(h_s))
            h_t_ = F.relu(tgt_layer(h_t))
            h_s = h_s_ + cross(h_t_)   # 跨连接
            h_t = h_t_ + cross(h_s_)   # 反向跨连接

        if domain == "source":
            return self.source_out(h_s).view(-1)
        else:
            return self.target_out(h_t).view(-1)

    def calculate_loss(self, interaction, domain="target"):
        user = interaction[self.USER_ID[domain]]
        item = interaction[self.ITEM_ID[domain]]
        label = interaction[self.LABEL[domain]]
        pred = self.forward(user, item, domain)
        return F.binary_cross_entropy_with_logits(pred, label.float())

    def predict(self, interaction, domain="target"):
        user = interaction[self.USER_ID[domain]]
        item = interaction[self.ITEM_ID[domain]]
        return torch.sigmoid(self.forward(user, item, domain))
