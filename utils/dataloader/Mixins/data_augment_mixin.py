"""
Data Augment Mixins only provide augment frameworks, specific methods should be applied dynamically
"""

from abc import ABC
from utils.dataloader.functional import *


class DataAugmentMixin(ABC):
    """Interface"""
    def __init__(self):
        pass


class SequentialDataAugmentMixin(DataAugmentMixin):
    pass