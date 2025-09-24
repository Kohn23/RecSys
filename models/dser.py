"""
DSER
################################################

Reference:
    Minsung Hong et al. " DSER:Deep-Sequential Embedding for single domain Recommendation."
    in Expert Systems With Applications 2018.

Note:
    This model has two separate parts: Doc2Vec and GMF + MLP
    It handles sequential information only when training Doc2Vec, thus the model is a general recommender

"""


import torch
from torch import nn

from recbole.model.abstract_recommender import GeneralRecommender


class DSER(GeneralRecommender):
    """DSER: Deep-Sequential Embedding for single domain Recommendation"""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.dataset = dataset
        self.LABEL = config["LABEL_FIELD"]

        # Doc2Vec parameters
        self.doc2vec_vector_size = config['doc2vec_vector_size']
        self.doc2vec_window = config['doc2vec_window']
        self.doc2vec_min_count = config['doc2vec_min_count']
        self.doc2vec_epochs = config['doc2vec_epochs']
        self.doc2vec_dm = config['doc2vec_dm']  # 1 for PVDM, 0 for PVDBOW

        # Model parameters
        self.embedding_size = config['embedding_size']
        self.mlp_layers = config['mlp_layers']
        self.dropout_prob = config['dropout_prob']

        # Model components
        self.user_embedding = nn.Embedding(dataset.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(dataset.item_num, self.embedding_size)

        # GMF component
        self.gmf_output = nn.Linear(self.embedding_size, 1)

        # MLP component
        mlp_input_size = self.doc2vec_vector_size * 2  # user_vec + item_vec from Doc2Vec
        self.mlp_layers_list = nn.ModuleList()
        input_size = mlp_input_size

        for output_size in self.mlp_layers:
            self.mlp_layers_list.append(nn.Linear(input_size, output_size))
            self.mlp_layers_list.append(nn.ReLU())
            self.mlp_layers_list.append(nn.Dropout(self.dropout_prob))
            input_size = output_size

        self.mlp_output = nn.Linear(input_size, 1)

        # Final fusion layer
        self.fusion_layer = nn.Linear(2, 1)  # Combine GMF and MLP outputs

        # Doc2Vec models and embeddings (will be initialized during training)
        self.user_doc2vec = None
        self.item_doc2vec = None
        self.user_doc2vec_embeddings = None
        self.item_doc2vec_embeddings = None
        self.requires_doc2vec_training = True

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _save_doc2vec_embeddings(self, dataset):
        """Save Doc2Vec embeddings as tensors"""
        n_users = self.n_users
        n_items = self.n_items

        # User embeddings
        self.user_doc2vec_embeddings = torch.zeros(n_users, self.doc2vec_vector_size)
        for uid in range(n_users):
            try:
                embedding = self.user_doc2vec.dv[f'user_{uid}']
                self.user_doc2vec_embeddings[uid] = torch.FloatTensor(embedding)
            except KeyError:
                self.user_doc2vec_embeddings[uid] = torch.randn(self.doc2vec_vector_size) * 0.1

        # Item embeddings
        self.item_doc2vec_embeddings = torch.zeros(n_items, self.doc2vec_vector_size)
        for iid in range(n_items):
            try:
                embedding = self.item_doc2vec.dv[f'item_{iid}']
                self.item_doc2vec_embeddings[iid] = torch.FloatTensor(embedding)
            except KeyError:
                self.item_doc2vec_embeddings[iid] = torch.randn(self.doc2vec_vector_size) * 0.1

        # Move to device
        if torch.cuda.is_available():
            self.user_doc2vec_embeddings = self.user_doc2vec_embeddings.cuda()
            self.item_doc2vec_embeddings = self.item_doc2vec_embeddings.cuda()

    def get_doc2vec_embeddings(self, user_ids, item_ids):
        """Get Doc2Vec embeddings for given user and item IDs"""
        if self.user_doc2vec_embeddings is None or self.item_doc2vec_embeddings is None:
            raise ValueError("Doc2Vec embeddings not initialized. Please train Doc2Vec first.")

        user_embs = self.user_doc2vec_embeddings[user_ids]
        item_embs = self.item_doc2vec_embeddings[item_ids]
        return user_embs, item_embs

    def forward(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        # Get Doc2Vec embeddings
        user_doc2vec_emb, item_doc2vec_emb = self.get_doc2vec_embeddings(user, item)

        # GMF pathway
        user_gmf_emb = self.user_embedding(user)
        item_gmf_emb = self.item_embedding(item)
        gmf_interaction = user_gmf_emb * item_gmf_emb  # Element-wise product
        gmf_output = torch.sigmoid(self.gmf_output(gmf_interaction))

        # MLP pathway with Doc2Vec embeddings
        mlp_input = torch.cat([user_doc2vec_emb, item_doc2vec_emb], dim=1)

        for layer in self.mlp_layers_list:
            mlp_input = layer(mlp_input)

        mlp_output = torch.sigmoid(self.mlp_output(mlp_input))

        # Fusion of GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        final_output = torch.sigmoid(self.fusion_layer(combined))

        return final_output.squeeze()

    def calculate_loss(self, interaction):
        """Calculate binary cross-entropy loss with neg-sampling"""
        pos_scores = self.forward(interaction)
        pos_labels = torch.ones_like(pos_scores)

        batch_size = interaction[self.USER_ID].shape[0]
        neg_items = torch.randint(0, self.n_items, (batch_size * 4,),
                                  device=interaction[self.USER_ID].device)

        neg_interaction = {}
        for field in interaction:
            if field == self.ITEM_ID:

                users = interaction[self.USER_ID]
                expanded_users = users.repeat_interleave(4)
                neg_interaction[self.USER_ID] = expanded_users
                neg_interaction[self.ITEM_ID] = neg_items
            else:
                neg_interaction[field] = interaction[field].repeat_interleave(4)

        neg_scores = self.forward(neg_interaction)
        neg_labels = torch.zeros_like(neg_scores)

        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])

        loss_fn = nn.BCELoss()
        return loss_fn(all_scores, all_labels)

    def predict(self, interaction):
        """Predict scores for given interactions"""
        return self.forward(interaction)

    def full_sort_predict(self, interaction):
        """Full sort prediction for evaluation"""
        user = interaction[self.USER_ID]

        user_emb = self.user_embedding(user).unsqueeze(1)  # (batch_size, 1, emb_dim)
        all_item_emb = self.item_embedding.weight.unsqueeze(0)  # (1, item_num, emb_dim)

        # GMF scores
        gmf_scores = torch.matmul(user_emb, all_item_emb.permute(0, 2, 1)).squeeze(1)  # (batch_size, item_num)

        # For MLP pathway, we need to get Doc2Vec embeddings for all items
        if self.user_doc2vec_embeddings is not None and self.item_doc2vec_embeddings is not None:
            user_doc2vec_emb = self.user_doc2vec_embeddings[user].unsqueeze(1)  # (batch_size, 1, doc2vec_dim)
            all_item_doc2vec_emb = self.item_doc2vec_embeddings.unsqueeze(0)  # (1, item_num, doc2vec_dim)

            # Expand to batch size
            user_doc2vec_expanded = user_doc2vec_emb.expand(-1, all_item_doc2vec_emb.shape[1], -1)
            mlp_input = torch.cat([user_doc2vec_expanded, all_item_doc2vec_emb], dim=2)

            # Process through MLP layers
            original_shape = mlp_input.shape
            mlp_input = mlp_input.reshape(-1, mlp_input.shape[-1])

            for layer in self.mlp_layers_list:
                mlp_input = layer(mlp_input)

            mlp_scores = self.mlp_output(mlp_input).reshape(original_shape[0], original_shape[1])
            mlp_scores = torch.sigmoid(mlp_scores)

            # Combine GMF and MLP scores
            combined_scores = torch.cat([
                gmf_scores.unsqueeze(2),
                mlp_scores.unsqueeze(2)
            ], dim=2)

            final_scores = torch.sigmoid(self.fusion_layer(combined_scores)).squeeze(2)
            return final_scores
        else:
            return torch.sigmoid(gmf_scores)