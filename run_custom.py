# general
import os
import sys
import torch.nn as nn
from typing import Type
from logging import getLogger

# configs
from config.config_dicts import config_cdr, config_sr

# single-domain
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.data.transform import construct_transform
from recbole.utils import init_logger, init_seed, set_color, get_flops

from models import SASRecInfoNCE, DSER
from trainers import DSERTrainer


def run_single_domain(module: Type[nn.Module], trainer, dataset, config_dict):
    config = Config(model=module, dataset=dataset, config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # data
    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model
    model = module(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # this will run forward() and unfortunately it is incompatible with some embeddings methods
    # transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # train and eval
    train = trainer(config, model)
    best_valid_score, best_valid_result = train.fit(train_data, valid_data, show_progress=True)
    test_result = train.evaluate(test_data)

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return test_result


# cross-domain
from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset as create_dataset_cdr
from recbole_cdr.data import data_preparation as data_preparation_cdr
from recbole_cdr.trainer import CrossDomainTrainer


if __name__ == "__main__":
    # debug
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    run_single_domain(module=SASRecInfoNCE, trainer=Trainer, dataset='afo_food', config_dict=config_sr)
    # run_single_domain(module=DSER, trainer=DSERTrainer, dataset='afo_office', config_dict=config_sr)
