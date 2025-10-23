"""
    RecBole-GNN based layers
"""


import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul


class LightGCNConv(MessagePassing):
    def __init__(self, dim):
        super(LightGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BipartiteGCNConv(MessagePassing):
    def __init__(self, dim):
        super(BipartiteGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.lin2 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(torch.mul(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

        self._reset_parameters()

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class SequenceGraphEncoder(nn.Module):
    """GNN-based Sequence Encoder based on the description of SimDCL"""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        super(SequenceGraphEncoder, self).__init__()

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            SRGNNCell(hidden_size) for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def build_graph_from_sequence(self, item_sequence, item_embeddings):
        """
        Build directed graph from item sequence with normalized edge weights

        Args:
            item_sequence: [seq_len] - sequence of item indices
            item_embeddings: nn.Embedding - item embedding layer

        Returns:
            x: [num_unique_nodes, hidden_size] - node embeddings
            edge_index: [2, num_edges] - graph edges
            edge_weight: [num_edges] - normalized edge weights
            alias_inputs: mapping from sequence position to node index
        """
        # Get unique nodes and create mapping
        unique_nodes, inverse_indices = torch.unique(item_sequence, return_inverse=True)
        num_unique = len(unique_nodes)

        # Create alias mapping: sequence position -> node index
        alias_inputs = inverse_indices

        # Get node embeddings
        x = item_embeddings(unique_nodes)  # [num_unique, hidden_size]

        # Build edge list with counts
        edge_dict = {}
        for i in range(1, len(item_sequence)):
            src = alias_inputs[i - 1].item()
            dst = alias_inputs[i].item()
            edge = (src, dst)
            edge_dict[edge] = edge_dict.get(edge, 0) + 1

        # Convert to edge_index and edge_weight
        edge_list = list(edge_dict.keys())
        if not edge_list:
            # If no edges, create self-loops
            edge_index = torch.tensor([[i for i in range(num_unique)],
                                       [i for i in range(num_unique)]],
                                      dtype=torch.long)
            edge_weight = torch.ones(num_unique) / num_unique
        else:
            src_nodes, dst_nodes = zip(*edge_list)
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

            # Compute normalized weights: count / out_degree
            edge_weight = torch.tensor([edge_dict[edge] for edge in edge_list], dtype=torch.float)

            # Normalize by out-degree
            out_degree = torch.zeros(num_unique)
            for src, count in zip(src_nodes, edge_weight):
                out_degree[src] += count

            for i, (src, _) in enumerate(edge_list):
                if out_degree[src] > 0:
                    edge_weight[i] = edge_weight[i] / out_degree[src]

        return x, edge_index, edge_weight, alias_inputs

    def forward(self, item_sequence, item_embeddings):
        """
        Args:
            item_sequence: [seq_len] - input item sequence
            item_embeddings: nn.Embedding - item embedding layer

        Returns:
            sequence_representation: [hidden_size] - encoded sequence representation
            node_representations: [num_unique_nodes, hidden_size] - updated node embeddings
        """
        # Build graph from sequence
        x, edge_index, edge_weight, alias_inputs = self.build_graph_from_sequence(
            item_sequence, item_embeddings
        )

        device = item_sequence.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        # Apply GNN layers
        hidden = x
        for gnn_layer in self.gnn_layers:
            hidden = gnn_layer(hidden, edge_index, edge_weight)
            hidden = self.layer_norm(hidden)
            hidden = self.dropout_layer(hidden)

        # Get sequence representation (using last node in sequence)
        # You can modify this aggregation strategy as needed
        seq_nodes = hidden[alias_inputs]  # [seq_len, hidden_size]
        sequence_representation = seq_nodes[-1]  # Use last node as sequence representation

        return sequence_representation, hidden
