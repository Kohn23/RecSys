import os
import pickle
from logging import getLogger
from recbole.utils import set_color
from recbole.utils.argument_list import dataset_arguments

from utils import CrossDomainModelType


def create_dataset_cdr(config):
    """
    This is a fixing of create_dataset from recbole_cdr
    """
    # dataset_module = importlib.import_module('recbole_cdr.data.dataset')
    # fixing
    from utils.dataset import cross_dmain_datasets as dataset_module

    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    else:
        model_type = config['MODEL_TYPE']
        type2class = {
            CrossDomainModelType.GENERAL: 'CrossDomainDatasetFixing'
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset

    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset