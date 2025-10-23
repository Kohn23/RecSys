# general
import os
import sys
import torch.nn as nn
from typing import Type
from logging import getLogger

from recbole.utils import init_logger, init_seed, set_color
from utils.logger import *


# single-domain
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from utils.trainer import DSERTrainer
from models import SASRecInfoNCE, DSER, CLF4SRec


def run_single_domain(module: Type[nn.Module], trainer, dataset, config_file_list):

    config = Config(model=module, dataset=dataset, config_file_list=config_file_list)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # preprocess
    dataset = create_dataset(config)
    logger.info(dataset)

    # TODO: find a way to unify creating dataloader
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
# from config.configurators import CDRConfigFixing
# from utils.dataset import create_dataset_cdr
# from utils.dataloader import data_preparation_cdr
#
# # dataset settings are in config_dict
# def run_cross_domain(module: Type[nn.Module], trainer, config_dict):
#     config = CDRConfigFixing(model=module, config_dict=config_dict)
#
#     init_seed(config['seed'], config['reproducibility'])
#     # logger initialization
#     init_logger_cdr(config)
#     logger = getLogger()
#     logger.info(config)
#
#     # dataset filtering
#     dataset = create_dataset_cdr(config)
#     logger.info(dataset)
#     # dataset splitting
#     train_data, valid_data, test_data = data_preparation_cdr(config, dataset)
#
#     # model loading and initialization
#     init_seed(config['seed'], config['reproducibility'])
#     model = module(config, train_data.dataset).to(config['device'])
#     logger.info(model)
#     # trainer loading and initialization
#     trainer = trainer(config, model)
#
#     # model training
#     best_valid_score, best_valid_result = trainer.fit(
#         train_data, valid_data, show_progress=config['show_progress']
#     )
#
#     # model evaluation
#     test_result = trainer.evaluate(test_data)
#
#     logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
#     logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == "__main__":
    # debug
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    config_file_list = [
        './config/properties/overall.yaml',
        './config/properties/dataset/single_domain.yaml',
        './config/properties/model/DSER.yaml',
    ]
    run_single_domain(module=DSER, trainer=DSERTrainer, dataset='abe_electronics', config_file_list=config_file_list)

    # run_single_domain(module=DSER, trainer=DSERTrainer, dataset='abe_23_beauty_and_pc', config_dict=config_sr)
    # run_cross_domain(module=DTCDR, trainer=CrossDomainTrainer, config_dict=config_cdr)
