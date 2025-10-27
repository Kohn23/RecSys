"""
Data Augment Mixins only provide augment frameworks, specific methods should be applied dynamically
"""

import random
import torch
from recbole.data.interaction import Interaction
from abc import ABC, abstractmethod
from utils.dataloader.functional import (
    seq_item_crop,
    seq_item_mask,
    seq_item_noise,
    seq_item_reorder
)

class DataAugmentMixin(ABC):
    """Interface"""
    def __init__(self, config):
        pass

    @abstractmethod
    def augment(self, interaction):
        """data augment"""
        pass


class SequentialDataAugmentMixin(DataAugmentMixin):
    def __init__(self, config):
        super().__init__(config)
        self.augment_method = self._get_augment_methods(config)


        self.crop_eta = config["eta"]
        self.mask_gamma = config["gamma"]
        self.noise_r = config["noise_r"]
        self.reorder_beta = config.get["beta"]

    def _set_fields(self, config):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]

    def _get_method(self, config):
        model = config['model'].lower()

        get_methods = {
            'cl4srec': ['crop', 'mask', 'reorder'],
        }

        augment_methods = None
        if model in get_methods:
            augment_methods = get_methods[model]
        elif config['augment']:
            augment_methods = config['augment']

        if not augment_methods:
            return []

        available_methods = {
            'crop': seq_item_crop,
            'mask': seq_item_mask,
            'reorder': seq_item_reorder,
            'noise': seq_item_noise
        }

        enabled_methods = []
        for method_name in augment_methods:
            if method_name in available_methods:
                enabled_methods.append(available_methods[method_name])
                print(f"Loaded augmenter: {method_name}")
            else:
                print(f"Warning: Unknown augmentation method '{method_name}'")

        return enabled_methods

    def

    def augment(self, interaction):

        return interaction
