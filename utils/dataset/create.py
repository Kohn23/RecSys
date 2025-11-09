import importlib
import os
import pickle
from logging import getLogger

from recbole.utils import set_color
from utils import ExtModelType


def _find_custom_dataset_class(config):
    model_name = config["model"]

    custom_modules = [
        "utils.dataset.graph_datasets",
        # "utils.dataset.crossdomain_datasets",
    ]

    if "custom_dataset_modules" in config:
        custom_modules.extend(config["custom_dataset_modules"])

    for module_path in custom_modules:
        try:
            module = importlib.import_module(module_path)
            class_name = f"{model_name}Dataset"
            if hasattr(module, class_name):
                return getattr(module, class_name)
            else:
                model_type = config["MODEL_TYPE"]
                type2class = {
                    ExtModelType.GRAPH: "GeneralGraphDataset",
                    ExtModelType.SESSION_GRAPH: "SessionGraphDataset"
                }
                if model_type in type2class:
                    class_name = type2class[model_type]
                    if hasattr(module, class_name):
                        return getattr(module, class_name)

        # TODO: Maybe not the best way to handle
        except ImportError:
            continue

    return None


def _create_dataset_with_caching(config, dataset_class):
    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file

    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        if _validate_dataset_config(dataset, config):
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def _validate_dataset_config(dataset, config):
    from recbole.utils.argument_list import dataset_arguments

    dataset_args_unchanged = True
    for arg in dataset_arguments + ["seed", "repeatable"]:
        if config[arg] != dataset.config[arg]:
            dataset_args_unchanged = False
            break
    return dataset_args_unchanged


def create_dataset(config):
    """a solution from recbole-gnn"""

    dataset_class = _find_custom_dataset_class(config)

    if dataset_class is not None:
        return _create_dataset_with_caching(config, dataset_class)
    else:
        from recbole.data import create_dataset as recbole_create_dataset
        return recbole_create_dataset(config)

