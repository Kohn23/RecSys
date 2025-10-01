"""
SimDCL
################################################

Reference:
    Zuxiang Xie, Junyi Li. "  Simple Debiased Contrastive Learning for Sequential Recommendation."
    in Knowledge Based Systems 2024.

Note:
    This model has two separate parts: Doc2Vec and GMF + MLP
    It handles sequential information only when training Doc2Vec, thus the model is a general recommender

"""

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss, EmbLoss

from models.layers import SequenceGraphEncoder


class SimDCL(SequentialRecommender):
    """SimDCL: Simple Debiased Contrastive Learning for Sequential Recommendation"""
    def __init__(self, config, dataset):
        super(SimDCL, self).__init__(config, dataset)

        # TransformerEncoder
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # d
        self.inner_size = config['inner_size']    # 4d
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        # noise
        self.noise_ratio = config['noise_ratio']  # K
        self.filter_threshold = config['filter_threshold']  # φ
        self.temperature = config['temperature']  # τ
        self.gradient_step = config['gradient_step']  # t
        self.gradient_lr = config['gradient_lr']  # μ

        self.lambda_t = config['lambda_t']  # λ1
        self.lambda_g = config['lambda_g']  # λ2

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.transformer_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.gnn_encoder = SequenceGraphEncoder(
            hidden_size=self.hidden_size,
            num_layers=self.n_layers
        )

        self.loss_fct = nn.CrossEntropyLoss()
        self.contrastive_loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, item_seq, item_seq_len):

        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_emb = self.position_embedding(position_ids)
        input_emb = item_emb + position_emb

        transformer_output = self.transformer_encoder(input_emb, output_all_encoded_layers=True)
        transformer_output = transformer_output[-1]

        gnn_output = self.gnn_encoder(item_seq, item_emb)

        return transformer_output, gnn_output

    def generate_noise_augmentation(self, seq_emb):
        noise = torch.randn_like(seq_emb) * self.noise_ratio
        augmented_emb = seq_emb + noise
        return augmented_emb

    def gradient_update_noise_samples(self, original_emb, augmented_emb):
        # formula(10)
        for _ in range(self.gradient_step):
            augmented_emb.requires_grad_(True)
            loss = self.compute_non_uniformity_loss(original_emb, augmented_emb)
            grad = torch.autograd.grad(loss, augmented_emb, retain_graph=True)[0]
            augmented_emb = augmented_emb + self.gradient_lr * grad / (grad.norm(2, dim=-1, keepdim=True) + 1e-8)
            augmented_emb = augmented_emb.detach()
        return augmented_emb

    def compute_non_uniformity_loss(self, orig_emb, aug_emb):
        # formula(9)
        sim_matrix = torch.matmul(orig_emb, aug_emb.T) / self.temperature
        labels = torch.arange(orig_emb.size(0)).to(orig_emb.device)
        loss = self.contrastive_loss_fct(sim_matrix, labels)
        return loss

    def object_filtering(self, orig_emb, pos_emb, neg_emb):
        # # formula(11)
        sim_pos = torch.cosine_similarity(orig_emb.unsqueeze(1), pos_emb.unsqueeze(0), dim=-1).mean(dim=1)
        sim_neg = torch.cosine_similarity(orig_emb.unsqueeze(1), neg_emb.unsqueeze(0), dim=-1).mean(dim=1)

        mask_pos = (sim_pos >= self.filter_threshold).float()
        mask_neg = (sim_neg < self.filter_threshold).float()

        return mask_pos, mask_neg

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        trans_output, gnn_output = self.forward(item_seq, item_seq_len)
        trans_scores = self.predict(trans_output, trans_output[:, -1, :])
        gnn_scores = self.predict(gnn_output, gnn_output[:, -1, :])

        loss_trans = self.loss_fct(trans_scores, pos_items)
        loss_gnn = self.loss_fct(gnn_scores, pos_items)
        loss_rec = self.lambda_t * loss_trans + self.lambda_g * loss_gnn

        aug_trans = self.generate_noise_augmentation(trans_output)
        aug_trans = self.gradient_update_noise_samples(trans_output, aug_trans)

        aug_gnn = self.generate_noise_augmentation(gnn_output)
        aug_gnn = self.gradient_update_noise_samples(gnn_output, aug_gnn)

        mask_trans_pos, mask_trans_neg = self.object_filtering(trans_output, aug_trans, aug_trans)
        mask_gnn_pos, mask_gnn_neg = self.object_filtering(gnn_output, aug_gnn, aug_gnn)

        # 对比损失（简化版）
        loss_cl_trans = self.compute_non_uniformity_loss(trans_output, aug_trans)
        loss_cl_gnn = self.compute_non_uniformity_loss(gnn_output, aug_gnn)

        total_loss = loss_rec + self.lambda_t * loss_cl_trans + self.lambda_g * loss_cl_gnn

        return total_loss

    def predict(self, sequence_output, target_emb):
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(target_emb, test_item_emb.T)
        return scores
