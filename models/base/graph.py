from recbole.model.abstract_recommender import GeneralRecommender, SequentialRecommender
from utils import ExtModelType


class GeneralGraphRecommender(GeneralRecommender):
    """
    This is an abstract general graph recommender. All the general graph models should implement in this class.
    The base general graph recommender class provide the basic U-I graph dataset and parameters information.
    """
    type = ExtModelType.GRAPH

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat(enable_sparse=config["enable_sparse"])
        self.use_sparse = config["enable_sparse"] and dataset.is_sparse
        if self.use_sparse:
            self.edge_index, self.edge_weight = self.edge_index.to(self.device), None
        else:
            self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)


class SessGraphRecommender(SequentialRecommender):
    """
    The base for Session Graph Recommender: Override type
    """
    type = ExtModelType.SESSION_GRAPH

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
