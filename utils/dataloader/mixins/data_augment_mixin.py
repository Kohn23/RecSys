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

    map_function = {
        'crop': seq_item_crop,
        'mask': seq_item_mask,
        'reorder': seq_item_reorder,
        #'noise': seq_item_noise
    }

    def __init__(self, config):
        pass

    @abstractmethod
    def augment(self, interaction):
        """data augment"""
        pass


class SequentialDataAugmentMixin(DataAugmentMixin):
    """
    Sequential augment
    """

    get_preset = {
        'cl4srec': {
            'func': ['crop', 'mask', 'reorder'],
            'mode': 'dual-view'
        },
    }

    def __init__(self, config):
        super().__init__(config)
        self.augment_method = self._get_method(config)

    def _set_params(self, config, augment_method):
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.mode = augment_method['mode']

        if 'crop' in augment_method['func']:
            self.CROP_ITEM_SEQ = "Crop_" + self.ITEM_SEQ
            self.CROP_ITEM_SEQ_LEN = "Crop_" + self.ITEM_SEQ_LEN
            config["CROP_ITEM_SEQ"] = self.CROP_ITEM_SEQ
            config["CROP_ITEM_SEQ_LEN"] = self.CROP_ITEM_SEQ_LEN
            # params
            self.crop_eta = config["eta"]

        if 'mask' in augment_method['func']:
            self.MASK_ITEM_SEQ = "Mask_" + self.ITEM_SEQ
            self.MASK_ITEM_SEQ_LEN = "Mask_" + self.ITEM_SEQ_LEN
            config["MASK_ITEM_SEQ"] = self.MASK_ITEM_SEQ
            config["MASK_ITEM_SEQ_LEN"] = self.MASK_ITEM_SEQ_LEN
            # params
            self.mask_gamma = config["gamma"]

        if 'reorder' in augment_method['func']:
            self.REORDER_ITEM_SEQ = "Reorder_" + self.ITEM_SEQ
            self.REORDER_ITEM_SEQ_LEN = "Reorder_" + self.ITEM_SEQ_LEN
            config["REORDER_ITEM_SEQ"] = self.REORDER_ITEM_SEQ
            config['REORDER_ITEM_SEQ_LEN'] = self.REORDER_ITEM_SEQ_LEN
            # params
            self.reorder_beta = config.get["beta"]

    def _get_method(self, config):
        model = config['model'].lower()

        augment_method = None
        if model in self.get_preset:
            augment_method = self.get_preset[model]
        elif config['augment']:
            augment_method = config['augment']

        if not augment_method:
            return []

        self._set_params(config, augment_method)

        return augment_method

    def _get_single_view(self, seq, length):

    def _get_views(self, seq, length):


    def augment(self, interaction):
        if 


        return interaction
