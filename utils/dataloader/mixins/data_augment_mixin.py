"""
Data Augment Mixins only provide augment frameworks, specific methods should be applied dynamically
"""

import math
import random
import torch
import numpy as np
from recbole.data.interaction import Interaction
from abc import ABC, abstractmethod


class DataAugmentMixin(ABC):
    """Interface"""
    def __init__(self, config):
        pass

    @abstractmethod
    def augment(self, interaction):
        """data augment"""
        pass


class CropItemSequence(DataAugmentMixin):
    """Random crop for item sequence."""
    def __init__(self, config):
        super().__init__(config)
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.CROP_ITEM_SEQ = "Crop_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.CROP_ITEM_SEQ_LEN = self.CROP_ITEM_SEQ + self.ITEM_SEQ_LEN
        self.crop_eta = config["eta"]
        config["CROP_ITEM_SEQ"] = self.CROP_ITEM_SEQ
        config["CROP_ITEM_SEQ_LEN"] = self.CROP_ITEM_SEQ_LEN

    def augment(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        crop_item_seq_list, crop_item_seqlen_list = [], []

        for seq, length in zip(item_seq, item_seq_len):
            crop_len = math.floor(length * self.crop_eta)
            if crop_len == 0:
                crop_item_seq_list.append(
                    torch.tensor(seq, dtype=torch.long, device=device)
                )
                crop_item_seqlen_list.append(length)
                continue
            crop_begin = random.randint(0, length - crop_len)
            crop_item_seq = np.zeros(seq.shape[0])
            if crop_begin + crop_len < seq.shape[0]:
                crop_item_seq[:crop_len] = seq[crop_begin: crop_begin + crop_len]
            else:
                crop_item_seq[:crop_len] = seq[crop_begin:]
            crop_item_seq_list.append(
                torch.tensor(crop_item_seq, dtype=torch.long, device=device)
            )
            crop_item_seqlen_list.append(
                torch.tensor(crop_len, dtype=torch.long, device=device)
            )
        new_dict = {
            self.CROP_ITEM_SEQ: torch.stack(crop_item_seq_list),
            self.CROP_ITEM_SEQ_LEN: torch.stack(crop_item_seqlen_list),
        }
        interaction.update(Interaction(new_dict))
        return interaction


class ReorderItemSequence(DataAugmentMixin):
    """Reorder operation for item sequence."""
    def __init__(self, config):
        super().__init__(config)
        self.ITEM_SEQ = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
        self.REORDER_ITEM_SEQ = "Reorder_" + self.ITEM_SEQ
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.REORDER_ITEM_SEQ_LEN = self.REORDER_ITEM_SEQ + self.ITEM_SEQ_LEN
        self.reorder_beta = config["beta"]
        config["REORDER_ITEM_SEQ"] = self.REORDER_ITEM_SEQ
        config["REORDER_ITEM_SEQ_LEN"] = self.REORDER_ITEM_SEQ_LEN

    def augment(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        device = item_seq.device
        reorder_seq_list, reorder_seqlen_list = [], []

        for seq, length in zip(item_seq, item_seq_len):
            reorder_len = math.floor(length * self.reorder_beta)
            reorder_begin = random.randint(0, length - reorder_len)
            reorder_item_seq = seq.cpu().detach().numpy().copy()

            shuffle_index = list(range(reorder_begin, reorder_begin + reorder_len))
            random.shuffle(shuffle_index)
            reorder_item_seq[reorder_begin : reorder_begin + reorder_len] = (
                reorder_item_seq[shuffle_index]
            )
            reorder_seqlen_list.append(length)
            reorder_seq_list.append(
                torch.tensor(reorder_item_seq, dtype=torch.long, device=device)
            )
        new_dict = {
            self.REORDER_ITEM_SEQ: torch.stack(reorder_seq_list),
            self.REORDER_ITEM_SEQ_LEN: torch.stack(reorder_seqlen_list)
        }
        interaction.update(Interaction(new_dict))
        return interaction


class SequentialDataAugmentMixin(DataAugmentMixin):
    def __init__(self, config):
        super().__init__(config)
        self.augmenters = self._build_augmenters(config)

    def _build_augmenters(self, config):
        augmenters = []
        model = config['model'].lower()

        get_model_methods = {
            'cl4srec':['crop','reorder'],
        }

        init_methods = {
            'crop': CropItemSequence,
            'reorder': ReorderItemSequence,
        }

        augment_methods = None
        if model in get_model_methods:
            augment_methods = get_model_methods[model]
        elif config['augment']:
            augment_methods = config['augment']

        if not augment_methods:
            return augmenters

        for method_name in augment_methods:
            if method_name in init_methods:
                augmenter = init_methods[method_name](config)
                augmenters.append(augmenter)
                print(f"Loaded augmenter: {method_name}")
            else:
                print(f"Warning: Unknown augmentation method '{method_name}'")

        return augmenters

    def augment(self, interaction):
        if not self.augmenters:
            return interaction

        for augmenter in self.augmenters:
            interaction = augmenter.augment(interaction)

        return interaction
