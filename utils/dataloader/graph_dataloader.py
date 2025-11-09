import numpy as np
from recbole.data.dataloader import NegSampleEvalDataLoader, FullSortEvalDataLoader
from utils.dataloader.functional import build_session_graph
from utils.dataloader.sequential_dataloader import SequentialDataLoader


class SessGraphTrainDataLoader(SequentialDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def collate_fn(self, index):
        index = np.array(index)
        interactions = self._dataset[index]
        interactions = build_session_graph(self.dataset, interactions)
        augmented_data = self.augment(interactions)

        return self._neg_sampling(augmented_data)


class SessGraphNegSampleEvalDataLoader(NegSampleEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        # override
        self.transform = build_session_graph


class SessGraphFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.transform = build_session_graph
