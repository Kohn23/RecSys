"""
Data Augment Mixins only provide augment frameworks, specific methods should be applied dynamically
"""

import random
import torch
from typing import Dict, Any, Optional
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
        self.augment_method: Optional[Dict[str, Any]] = self._get_method(config)

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
        if length <= 1:
            return seq.clone(), length.clone()

        available_funcs = self.augment_method['func']
        chosen_func = random.choice(available_funcs)
        augment_func = self.map_function[chosen_func]

        if chosen_func == 'crop':
            aug_seq, aug_len = augment_func(seq, length, self.crop_eta)
        elif chosen_func == 'mask':
            aug_seq, aug_len = augment_func(seq, length, self.mask_gamma)
        elif chosen_func == 'reorder':
            aug_seq, aug_len = augment_func(seq, length, self.reorder_beta)
        else:
            aug_seq, aug_len = augment_func(seq, length)

        return aug_seq, aug_len

    def _get_views(self, interaction):
        """views for contrastive learning"""
        seqs = interaction[self.ITEM_SEQ]
        lengths = interaction[self.ITEM_SEQ_LEN]
        batch_size = len(seqs)

        if self.mode == 'single-view':
            aug_seqs = []
            aug_lens = []

            for i in range(batch_size):
                seq = seqs[i]
                length = lengths[i]
                aug_seq, aug_len = self._get_single_view(seq, length)
                aug_seqs.append(aug_seq)
                aug_lens.append(aug_len)

            # 更新interaction
            interaction.update(Interaction({
                'aug_view': torch.stack(aug_seqs),
                'aug_len': torch.stack(aug_lens)
            }))
        elif self.mode == 'dual-view':
            aug_seqs1 = []
            aug_lens1 = []
            aug_seqs2 = []
            aug_lens2 = []

            for i in range(batch_size):
                seq = seqs[i]
                length = lengths[i]
                aug_seq1, aug_len1 = self._get_single_view(seq, length)
                aug_seq2, aug_len2 = self._get_single_view(seq, length)

                aug_seqs1.append(aug_seq1)
                aug_lens1.append(aug_len1)
                aug_seqs2.append(aug_seq2)
                aug_lens2.append(aug_len2)

            interaction.update(Interaction({
                'aug1': torch.stack(aug_seqs1),
                'aug_len1': torch.stack(aug_lens1),
                'aug2': torch.stack(aug_seqs2),
                'aug_len2': torch.stack(aug_lens2)
            }))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return interaction

    def augment(self, interaction):
        if not self.augment_method:
            return interaction

        # contrastive learning views
        if self.augment_method['mode'] == 'single-view' or self.augment_method['mode'] == 'dual-view':
            interaction = self._get_views(interaction)

        return interaction
