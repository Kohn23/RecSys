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
from recbole.model.loss import BPRLoss

class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        # Inside the function, a standard deviation (stdv) is first calculated, which is based on the inverse square of embedding_size.
        # Then, all the parameters of the model are iterated over, initializing the data (i.e., the weights) for each parameter to a value drawn from the uniform distribution, ranging from -stdv to stdv.
        # This initialization helps to set the parameters to small random values at the beginning of training for better model training and optimization.
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SimDCL(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    # 1
    def __init__(self, config, dataset):
        super(SimDCL, self).__init__(config, dataset)

        # load parameters info
        ## tf
        # Get the values of "n_layers" and "n_heads" from "config", representing the number of layers and attention heads in the model, respectively
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        # Get "hidden_size", which represents the dimensions of the hidden state in the model, also the same as the dimensions of the embedded vector, and "inner_size", which represents the internal dimensions in the feedforward neural network
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        # Obtain hyperparameters related to the model's hidden state, attention, activation function, and layer normalization
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        # Gets hyperparameters for model initialization range and loss type
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        # Gets the batch size at training time
        self.batch_size = config['train_batch_size']

        # define layers and loss
        # It is a simple lookup table that stores embedded vectors of fixed size dictionaries,
        # means that, given a number, the embedding layer can return the embedding vector corresponding to the number, the embedding vector reflects the semantic relationship between the symbols represented by each number
        # The input is a numbered list, and the output is a list of corresponding symbolic embedding vectors
        # This code creates an embedding layer that maps a discrete integer number to a continuous embedding vector. The specific explanation is as follows:
        # nn.Embedding: This is a class in PyTorch that is used to create the embedding layer.
        # self.n_items: This is the number of items to be represented by the embedding layer, which indicates how many different embedding vectors to be created in the embedding layer, usually corresponding to the total number of items or the number of categories.
        # self.hidden_size: This is the dimension of the embedding vector, which can also be called the dimension of the embedding space. It specifies the length of each embedded vector.
        # padding_idx=0: This is an optional parameter that specifies a special number for "fill".
        # In sequence data, you can use a number (usually 0) as a fill item, and the embedding layer creates an all-zero embedding vector for that number.
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        # LayerNorm,BatchNorm
        # These two lines of code initialize a LayerNorm layer and a Dropout layer for standardization and random deactivation operations, respectively
        # LayerNorm is used to standardize the input
        # The dropout is used to randomly zero neurons with a certain probability during training to reduce overfitting
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # Initialized a loss function "loss_fct" based on the set loss type
        # 'BPR', "BPRLoss" class
        # 'CE', then use the cross entropy loss in PyTorch "nn.CrossEntropyLoss()"
        # If the loss type is not one of these two, an unrealized error is thrown
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # dim: 0: cos similarity is calculated between column vectors of corresponding columns
        # 1(default): calculates the similarity between row vectors
        # -1: is to calculate the similarity between row vectors
        self.simf1 = nn.CosineSimilarity(dim=1)

        ## gnn
        self.embedding_size = config['embedding_size']
        self.step = config['step']
        # define layers and loss
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_1 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_2 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_3 = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_out = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)

        ## cl
        self.count_dict = {'idx': 0}
        # Initializes a cross entropy loss "loss_fct_ce" to calculate the cross entropy loss
        self.loss_fct_ce = nn.CrossEntropyLoss()


        # parameters initialization
        self.apply(self._init_weights)


    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    # 2
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # 4
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
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output  # [B H]

    def forward_gnn(self, item_seq, item_seq_len):
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        hidden = self.item_embedding(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.embedding_size)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)

        l1 = self.linear_1(ht).view(ht.size(0), 1, ht.size(1))
        l2 = self.linear_2(seq_hidden)

        alpha = self.linear_3(torch.sigmoid(l1 + l2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_out(torch.cat([a, ht], dim=1))
        return seq_output  # [B H]

    # 3
    def calculate_loss(self, interaction):
        # cal loss
        loss = torch.tensor(0.0).to(self.device)
        loss += self.config['t_weight'] if not self.config['t_weight'] else self.config['t_weight'] * self.rec_loss(interaction, self.forward)
        loss += self.config['g_weight'] if not self.config['g_weight'] else self.config['g_weight'] * self.rec_loss(interaction, self.forward_gnn)
        if self.config['open_cl']:
            loss += self.config['cl_weight'] if not self.config['cl_weight'] else self.config['cl_weight'] * self.cl_loss(interaction)

        return loss


    # main loss
    def rec_loss(self, interaction, forward):
        # item_seq
        item_seq = interaction[self.ITEM_SEQ]  # N(B) * L(sequence length), long truncation, not enough to fill 0
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # seq_output It is the result of passing the input sequence forward through the neural network. It is a tensor of shape N * D, where N is the batch size and D is the dimension of the output
        seq_output = forward(item_seq, item_seq_len)  # N * D(Dim) # N * D(Dim)
        # pos_items Represents a positive item associated with a given sequence.
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            # Calculate Bayesian Personalized Ranking (BPR) losses, using positive and negative items and embeddings to calculate scores, and then calculate losses
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type == 'CE'
            # 'CE'(cross entropy), then calculate the loss by embedding the item, using the softmax function
            # item_emb get all
            test_item_emb = self.item_embedding.weight
            # seq_output and all item_emb do similarity calculation # tensor.transpose exchange matrix of two dimensions, transpose(dim0, dim1)
            # torch.matmul is multiplication of tensor,
            # Input can be high dimensional. 2D is normal matrix multiplication like tensor.mm.
            # When there are multiple dimensions, the extra one dimension is brought out as batch, and the other parts do matrix multiplication
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss) or torch.isinf(loss):
            print("rec_loss:", loss)
            loss = 1e-8
        return loss

    # 4
    def cl_loss(self, interaction):
        loss = torch.tensor(0.0).to(self.device)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        for loss_func_temp in self.config['loss_func_temp']:
            loss_func_temp_arr = loss_func_temp.split('#')
            match loss_func_temp_arr[0]:
                case 'loss_1':  # debias cl
                    seq_output1 = self.forward_gnn_gn('gn', item_seq, item_seq_len)
                    seq_output2 = self.forward_gnn_gn('gn', item_seq, item_seq_len)
                    loss += self.d_cl_loss_1(seq_output1, seq_output2, temp=float(loss_func_temp_arr[1]), batch_size=item_seq_len.shape[0])
                case 'loss_11':  # debias cl
                    seq_output1 = self.d_forward_tf_gn('gn', item_seq, item_seq_len)
                    seq_output2 = self.d_forward_tf_gn('gn', item_seq, item_seq_len)
                    loss += self.d_cl_loss_1(seq_output1, seq_output2, temp=float(loss_func_temp_arr[1]), batch_size=item_seq_len.shape[0])
                case 'loss_2':  # cl
                    if len(loss_func_temp_arr) > 2 and 'tf' == loss_func_temp_arr[2]:
                        seq_output1 = self.forward_tf_gn('gn', item_seq, item_seq_len)
                        seq_output2 = self.forward_tf_gn('gn', item_seq, item_seq_len)
                    else:
                        seq_output1 = self.forward_gnn_gn('gn', item_seq, item_seq_len)
                        seq_output2 = self.forward_gnn_gn('gn', item_seq, item_seq_len)
                    loss += self.cl_loss_1(seq_output1, seq_output2, temp=float(loss_func_temp_arr[1]), batch_size=item_seq_len.shape[0])
                case 'loss_3':  # cl nd
                    item_seq_1, item_seq_len_1 = tool.node_dropout(item_seq, item_seq_len, dropout_rate=self.config['nd_rate'])
                    seq_output1 = self.forward_tf_gn('gn', item_seq_1, item_seq_len_1)
                    seq_output2 = self.forward_tf_gn('gn', item_seq_1, item_seq_len_1)
                    loss += self.cl_loss_1(seq_output1, seq_output2, temp=float(loss_func_temp_arr[1]), batch_size=item_seq_len.shape[0])
                case 'loss_4':  # debias cl
                    seq_output1 = self.forward_gnn_gn('gn', item_seq, item_seq_len)
                    seq_output2 = self.forward_gnn_gn('gn', item_seq, item_seq_len)
                    da_seq_output_arr = []
                    for i in range(self.config['sample_batch']):
                        seq_output11 = self.forward_tf_gn('gn', item_seq, item_seq_len)
                        da_seq_output_arr.append(seq_output11)
                    loss += self.d_cl_loss_2(seq_output1, seq_output2, da_seq_output_arr, temp=float(loss_func_temp_arr[1]), batch_size=item_seq_len.shape[0])
                case 'loss_44':  # debias cl
                    seq_output1 = self.d_forward_tf_gn('gn', item_seq, item_seq_len)
                    seq_output2 = self.d_forward_tf_gn('gn', item_seq, item_seq_len)
                    da_seq_output_arr = []
                    for i in range(self.config['sample_batch']):
                        seq_output11 = self.d_forward_tf_gn('gn', item_seq, item_seq_len)
                        da_seq_output_arr.append(seq_output11)
                    loss += self.d_cl_loss_22(seq_output1, seq_output2, da_seq_output_arr, temp=float(loss_func_temp_arr[1]), batch_size=item_seq_len.shape[0])

        if torch.isnan(loss) or torch.isinf(loss):
            print("cl_loss:", loss)
            loss = 1e-8

        return loss

    def forward_tf_gn(self, da_method, item_seq, item_seq_len):
        # Handle embedding and transformation of sequence data
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device) # 1*max_seq_length
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq) # batch_size*max_seq_length
        # position_ids = position_ids[:, torch.randperm(position_ids.size()[1])]  # add token shuffle
        position_embedding = self.position_embedding(position_ids) # batch_size*max_seq_length*hidden_size 1024*50*60
        item_emb = self.item_embedding(item_seq)
        # zero_emb = self.item_embedding(torch.tensor(0,device=item_seq.device))
        # Data enhancement processing: According to different data enhancement methods da_method, the item is embedded in item_emb for processing.
        # 'gn', Gaussian noise is added to item_emb;
        # 'gn_m', then multiplies item_emb with Gaussian noise element level;
        # 'negative' generates a Gaussian noise tensor with the same shape as item_emb and assigns it to item_emb
        if da_method == 'gn': # gn_a
            noise = tool.gaussian_noise(item_emb, self.config['noise_base'])
            mask1 = item_seq.gt(0).unsqueeze(dim=2)
            noise1 = noise * mask1
            item_emb = item_emb + noise1
        elif da_method == 'gn_m': # gn_m
            noise = tool.gaussian_noise(item_emb, self.config['noise_base'])
            mask1 = item_seq.gt(0).unsqueeze(dim=2)
            noise1 = noise * mask1
            item_emb = torch.mul(item_emb, 1 + noise1)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def d_forward_tf_gn(self, da_method, item_seq, item_seq_len):
        # Handles embedding and transformation of sequence data
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device) # 1*max_seq_length
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq) # batch_size*max_seq_length
        # position_ids = position_ids[:, torch.randperm(position_ids.size()[1])]  # add token shuffle
        position_emb = self.position_embedding(position_ids) # batch_size*max_seq_length*hidden_size 1024*50*60
        item_emb = self.item_embedding(item_seq)  # 1024*50*64
        # zero_emb = self.item_embedding(torch.tensor(0, device=item_seq.device)) # 64,

        # da
        noise = tool.gaussian_noise(item_emb, self.config['noise_base'])
        mask1 = item_seq.gt(0).unsqueeze(dim=2)
        noise1 = noise * mask1
        item_emb = item_emb + noise1

        # Calculate the average sim for each ie and sequence
        ie_sim_mean = torch.zeros(item_emb.shape[:2], device=item_seq.device) # 1024*50
        new_item_seq = torch.clone(item_seq) # 1024*50
        for i in range(len(item_emb)):
            ie_sim_mean_i = ie_sim_mean[i]
            new_item_seq_i = new_item_seq[i]

            ie1_i = item_emb[i][:item_seq_len[i]]
            for j in range(len(ie1_i)):
                ie1_i_j = ie1_i[j]
                ie_sim_mean_i[j] = torch.mean(self.simf1(ie1_i_j, ie1_i))

            phi_val = self.config['phi'] * torch.max(ie_sim_mean_i)
            weights = torch.where(ie_sim_mean_i < phi_val, 0, 1)
            # mask_weights = torch.eye(ie_sim_mean.size(0), device=self.device) - torch.diag_embed(torch.diag(weights))
            # weights = weights + mask_weights
            new_item_seq_i = new_item_seq_i * weights
            new_item_seq[i] = new_item_seq_i

        item_emb = self.item_embedding(new_item_seq)  # 1024*50*64
        mask1 = new_item_seq.gt(0).unsqueeze(dim=2)
        noise1 = noise * mask1
        item_emb = item_emb + noise1

        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def forward_gnn_gn(self, da_method, item_seq, item_seq_len):
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        hidden = self.item_embedding(items)
        if da_method == 'gn': # gn_a
            noise = tool.gaussian_noise(hidden, self.config['noise_base'])
            mask1 = item_seq.gt(0).unsqueeze(dim=2)
            noise1 = noise * mask1
            hidden = hidden + noise1
        elif da_method == 'gn_m': # gn_m
            noise = tool.gaussian_noise(hidden, self.config['noise_base'])
            mask1 = item_seq.gt(0).unsqueeze(dim=2)
            noise1 = noise * mask1
            hidden = torch.mul(hidden, 1 + noise1)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.embedding_size)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)

        l1 = self.linear_1(ht).view(ht.size(0), 1, ht.size(1))
        l2 = self.linear_2(seq_hidden)

        alpha = self.linear_3(torch.sigmoid(l1 + l2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_out(torch.cat([a, ht], dim=1))
        return seq_output  # [B H]

    # 5
    # CE
    def cl_loss_1(self, z_i, z_j, temp, batch_size):  # B * D    B * D  (batch_size * dim)
        # Positive sample: It is the inner product of itself and itself, or the product of the NTH row in z_i and the NTH row in z_j.
        # Negative sample: is the inner product of itself and itself, or the product of the zeroth row in z_i and the zeroth row in z_j.
        N = 2 * batch_size

        sim = tool.sim(z_i,z_j,temp,self.config['simf'])

        # Take the diagonal entries of the matrix
        sim_i_j = torch.diag(sim, batch_size)  # B*1 # The main diagonal must be moved B rows up to be a positive sample
        sim_j_i = torch.diag(sim, -batch_size)  # B*1 # The main diagonal must be moved B rows up to be a positive sample

        # The specified value becomes a diagonal matrix
        # torch.diag_embed()

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # build negative samples
        # mask
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_correlated_samples(batch_size=self.batch_size)
        negative_samples = sim[mask].reshape(N, -1)  # Drop the positive sample mask, that is, move the diagonal elements in line B up/down
        # Create a negative sample
        # noise_1 = DAUtil.gaussian_noise(negative_samples, 0.01)
        # negative_samples_1 = negative_samples+noise_1
        # noise_2 = DAUtil.gaussian_noise(negative_samples, 0.01)
        # negative_samples_2 = negative_samples + noise_2
        # negative_samples = torch.cat([negative_samples, negative_samples_1, negative_samples_2], dim=1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        loss = self.loss_fct_ce(logits+1e-8, labels)  # BPRLoss or CrossEntropyLoss(Cross entropy loss)
        if torch.isnan(loss) or torch.isinf(loss):
            print("cl_loss_1:", loss)
            loss = 1e-8
        return loss


    # 5
    # CE false negative
    def d_cl_loss_1(self, z_i, da_z_j, temp, batch_size):  # B * D    B * D  (batch_size * dim)
        # Positive sample: It is the inner product of itself and itself, or the product of the NTH row in z_i and the NTH row in z_j.
        # Negative sample: is the inner product of itself and itself, or the product of the zeroth row in z_i and the zeroth row in z_j.
        N = 2 * batch_size

        # build positive_samples
        sim = tool.sim(z_i, da_z_j,temp,self.config['simf'])  # 2B * 2B
        # Take the diagonal entries of the matrix
        sim_i_j = torch.diag(sim, batch_size)  # B*1 # Main diagonal, must move up B line, is a positive sample
        sim_j_i = torch.diag(sim, -batch_size)  # B*1 # Main diagonal, must move up B line, is a positive sample
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # mask negative sample elements with high similarity
        phi_val = self.config['phi'] * torch.max(sim)
        weights = torch.where(sim > phi_val, 0, 1)
        # count = 2048*2048 - torch.count_nonzero(weights)
        mask_weights = torch.eye(sim.size(0), device=self.device) - torch.diag_embed(torch.diag(weights))
        weights = weights + mask_weights
        # count = 2048*2048 - torch.count_nonzero(weights)
        # dot(.): *,mul
        # Matrix multiplication(x): mm(2d),matmul(>=2d)
        sim = sim * weights

        # # Take the diagonal entries of the matrix
        # sim_i_j = torch.diag(sim, batch_size)  # B*1
        # sim_j_i = torch.diag(sim, -batch_size)  # B*1 # Main diagonal, must move up B line, is a positive sample
        #
        # # The specified value becomes a diagonal matrix
        # # torch.diag_embed()
        # positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # build negative samples
        # mask
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_correlated_samples(batch_size=self.batch_size)
        negative_samples = sim[mask].reshape(N, -1)  # Drop the positive sample mask, that is, move the diagonal elements in line B up/down

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        loss = self.loss_fct_ce(logits+1e-8, labels)  # BPRLoss or CrossEntropyLoss(Cross entropy loss)
        if torch.isnan(loss) or torch.isinf(loss):
            print("d_cl_loss_1:", loss)
            loss = 1e-8
        return loss


    # 5
    # CE false negative
    def d_cl_loss_2(self, z_i, z_j, da_z_j_arr, temp, batch_size):  # B * D    B * D  (batch_size * dim)
        N = 2 * batch_size

        # pgd
        # 1. Calculate the similarity between two pairs of positive samples
        # The cosine similarity of the positive samples z1 and z2 is calculated and divided by the temperature parameter temp.
        # Similarly, the cosine similarity of z1_fix and z2_fix is calculated
        # Reference sim
        ref_sim = tool.sim(z_i,z_j,temp,self.config['simf'])
        # 2. Positive samples correspond to labels
        # Create a label tensor with an integer from 0 to cos_sim.size(0)-1. Transfer it to the same device as self.device
        # tag
        labels = torch.arange(ref_sim.size(0)).long().to(self.device)
        # Iterate pgd times to iterate the optimized strategy
        for _ in range(self.config['pgd']):
            for j in range(len(da_z_j_arr)):
                da_z_j = da_z_j_arr[j]
                # # cal loss
                # loss = self.cts_loss_1(z_i_1, z_j, temp, batch_size)
                # # cal grad
                # noise_grad = torch.autograd.grad(loss, z_j, retain_graph=True)[0]
                #
                # z_j = z_j + (noise_grad / torch.norm(noise_grad, dim=-1, keepdim=True)).mul_(1e-1) #1e-3
                # # noise = torch.clamp(noise, 0, 1.0)
                # z_j_arr[j] = torch.where(torch.isnan(z_j), torch.zeros_like(z_j), z_j)

                # Similarity between positive and negative cases
                cos_sim_1 = tool.sim(z_j,da_z_j,temp,self.config['simf'])
                # Fusion of positive and negative examples
                cos_sim_fused = torch.cat([ref_sim, cos_sim_1], 1)
                # Calculate the cos_sim after multiple similarity fusion, and calculate the corresponding loss accordingly
                # Calculate the loss
                loss1 = self.loss_fct_ce(cos_sim_fused, labels)
                # Gradient calculation
                noise_grad = torch.autograd.grad(loss1, da_z_j, retain_graph=True)[0]

                da_z_j = da_z_j + (noise_grad / torch.norm(noise_grad, dim=-1, keepdim=True)).mul_(1e-3) # 1e-3
                # noise = torch.clamp(noise, 0, 1.0)
                da_z_j_arr[j] = torch.where(torch.isnan(da_z_j), torch.zeros_like(da_z_j), da_z_j)

        # calculate arr sim and mask samples
        negative_samples_arr = []
        positive_samples = None
        for da_z_j in da_z_j_arr:
            sim = tool.sim(z_i,da_z_j,temp,self.config['simf'])

            # mask negative sample elements with high similarity
            phi_val = self.config['phi'] * torch.max(ref_sim)
            weights = torch.where(sim > phi_val, 0, 1)
            # count = 2048*2048 - torch.count_nonzero(weights)
            mask_weights = torch.eye(sim.size(0), device=self.device) - torch.diag_embed(torch.diag(weights))
            weights = weights + mask_weights
            # count = 2048*2048 - torch.count_nonzero(weights)
            sim = sim * weights

            # Take the diagonal entries of the matrix
            sim_i_j = torch.diag(sim, batch_size)  # B*1
            sim_j_i = torch.diag(sim, -batch_size)  # B*1

            # The specified value becomes a diagonal matrix
            # torch.diag_embed()
            if positive_samples is None:
                positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

            # build negative samples
            # mask
            if batch_size != self.batch_size:
                mask = self.mask_correlated_samples(batch_size)
            else:
                mask = self.mask_correlated_samples(batch_size=self.batch_size)
            negative_samples = sim[mask].reshape(N, -1)  # Drop the positive sample mask, that is, move the diagonal elements in line B up/down
            negative_samples_arr.append(negative_samples)

        # Create a negative sample
        negative_samples = torch.cat(negative_samples_arr, dim=1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        loss = self.loss_fct_ce(logits+1e-8, labels)  # BPRLoss or CrossEntropyLoss(Cross entropy loss)
        if torch.isnan(loss) or torch.isinf(loss) or loss == 0:
            print("d_cl_loss_2:", loss)
            loss = 1e-8
        return loss


    # 5
    # CE false negative
    def d_cl_loss_22(self, z_i, z_j, da_z_j_arr, temp, batch_size):  # B * D    B * D  (batch_size * dim)
        N = 2 * batch_size

        # pgd
        # 1. Calculate the similarity between two pairs of positive samples
        # The cosine similarity of the positive samples z1 and z2 is calculated and divided by the temperature parameter temp.
        # Similarly, the cosine similarity of z1_fix and z2_fix is calculated
        # Reference sim
        ref_sim = tool.sim(z_i,z_j,temp,self.config['simf'])
        # 2. Positive samples correspond to labels
        # Create a label tensor with an integer from 0 to cos_sim.size(0)-1. Transfer it to the same device as self.device
        # tag
        labels = torch.arange(ref_sim.size(0)).long().to(self.device)
        # Iterate pgd times to iterate the optimized strategy
        for _ in range(self.config['pgd']):
            for j in range(len(da_z_j_arr)):
                da_z_j = da_z_j_arr[j]

                # Similarity between positive and negative cases
                cos_sim_1 = tool.sim(z_j,da_z_j,temp,self.config['simf'])
                # Positive and negative examples merge
                cos_sim_fused = torch.cat([ref_sim, cos_sim_1], 1)
                # Calculate the cos_sim after multiple similarity fusion, and calculate the corresponding loss accordingly
                # Calculate the loss
                loss1 = self.loss_fct_ce(cos_sim_fused, labels)
                # Gradient calculation
                noise_grad = torch.autograd.grad(loss1, da_z_j, retain_graph=True)[0]

                da_z_j = da_z_j + (noise_grad / torch.norm(noise_grad, dim=-1, keepdim=True)).mul_(1e-3) # 1e-3
                # noise = torch.clamp(noise, 0, 1.0)
                da_z_j_arr[j] = torch.where(torch.isnan(da_z_j), torch.zeros_like(da_z_j), da_z_j)

        # calculate arr sim and mask samples
        negative_samples_arr = []
        positive_samples = None
        for da_z_j in da_z_j_arr:
            sim = tool.sim(z_i,da_z_j,temp,self.config['simf'])

            # Take the diagonal entries of the matrix
            sim_i_j = torch.diag(sim, batch_size)  # B*1
            sim_j_i = torch.diag(sim, -batch_size)  # B*1

            # The specified value becomes a diagonal matrix
            # torch.diag_embed()
            if positive_samples is None:
                positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

            # build negative samples
            # mask
            if batch_size != self.batch_size:
                mask = self.mask_correlated_samples(batch_size)
            else:
                mask = self.mask_correlated_samples(batch_size=self.batch_size)
            negative_samples = sim[mask].reshape(N, -1)
            negative_samples_arr.append(negative_samples)

        # Create a negative sample
        negative_samples = torch.cat(negative_samples_arr, dim=1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        loss = self.loss_fct_ce(logits+1e-8, labels)  # BPRLoss or CrossEntropyLoss
        if torch.isnan(loss) or torch.isinf(loss) or loss == 0:
            print("d_cl_loss_2:", loss)
            loss = 1e-8
        return loss


    def mask_correlated_samples(self, batch_size):
        # print("================= mask_correlated_samples(self, batch_size) ===================")
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # 6
    def predict(self, interaction):
        # print("================= predict(self, interaction) ===================")
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        # print("================= full_sort_predict(self, interaction) ===================")
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def fast_predict(self, interaction):
        # print('fast_predict: ***********')
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction["item_id_with_negs"]
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding(test_item)  # [B, num, H]
        scores = torch.matmul(seq_output.unsqueeze(
            1), test_item_emb.transpose(1, 2)).squeeze()
        return scores
