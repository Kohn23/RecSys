"""
An update on TrainDataloader from Recbole to support:
1. Sequential Data Augment

Note:
    1. Modification is based on the latest version(1.2.1)
"""


import numpy as np
import torch
from logging import getLogger
from utils.dataloader.abstract_dataloader import (
    AbstractDataLoader,
    NegSampleDataLoader,
)
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType

from utils.dataloader.batch_augment import SequentialAugment


class SequentialDataLoader(NegSampleDataLoader):
    """
    Modified :class:TrainDataloader for Sequential data augmentation

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=True):
        self.logger = getLogger()
        self._set_neg_sample_args(
            config, dataset, config["MODEL_INPUT_TYPE"], config["train_neg_sample_args"]
        )
        self.sample_size = len(dataset)
        self.augment = SequentialAugment(config)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        if self.neg_sample_args["distribution"] != "none":
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        interactions = self._dataset[index]
        # transformed_data = self.transform(interactions)
        augmented_data = self.augment(interactions)
        return self._neg_sampling(augmented_data)
