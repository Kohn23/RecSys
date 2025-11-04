import numpy as np
import torch



from utils.dataloader.mixins import SessionGraphTransformMixin
from utils.dataloader.sequential_dataloader import SequentialDataLoader

class SessGraphTrainDataLoader(SessionGraphTransformMixin, SequentialDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
    def collate_fn(self, index):
        index = np.array(index)
        interactions = self._dataset[index]
        interactions = self.transform(self._da)
        augmented_data = self.augment(interactions)

        return self._neg_sampling(augmented_data)


class GraphNegSampleEvalDataLoader(NegSampleEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)

    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                data_list.append(self._neg_sampling(self._dataset[index]))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return self.transform(self._dataset, cur_data), idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class GraphFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)
