from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``GENERAL``: General Recommendation
    - ``SEQUENTIAL``: Sequential Recommendation
    - ``CONTEXT``: Context-aware Recommendation
    - ``KNOWLEDGE``: Knowledge-based Recommendation
    """

    GENERAL = 1
    SEQUENTIAL = 2
    CONTEXT = 3
    KNOWLEDGE = 4
    TRADITIONAL = 5
    DECISIONTREE = 6
    GRAPH = 7
    SESSION_GRAPH = 8


class CrossDomainDataLoaderState(Enum):
    """States for Cross-domain DataLoader.

    - ``BOTH``: Return both data in source domain and target domain.
    - ``SOURCE``: Only return the data in source domain.
    - ``TARGET``: Only return the data in target domain.
    - ``OVERLAP``: Return the overlapped users or items.
    """

    BOTH = 1
    SOURCE = 2
    TARGET = 3
    OVERLAP = 4


train_mode2state = {'BOTH': CrossDomainDataLoaderState.BOTH,
                    'SOURCE': CrossDomainDataLoaderState.SOURCE,
                    'TARGET': CrossDomainDataLoaderState.TARGET,
                    'OVERLAP': CrossDomainDataLoaderState.OVERLAP}


class DataLoaderType(Enum):
    """Type of DataLoaders.

    - ``ORIGIN``: Original DataLoader
    - ``FULL``: DataLoader for full-sort evaluation
    - ``NEGSAMPLE``: DataLoader for negative sample evaluation
    """

    ORIGIN = 1
    FULL = 2
    NEGSAMPLE = 3