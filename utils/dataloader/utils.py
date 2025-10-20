from recbole.data.dataloader import NegSampleEvalDataLoader
from recbole.data.utils import load_split_dataloaders, save_split_dataloaders, create_samplers
from recbole.utils import set_color
from recbole_cdr.sampler import CrossDomainSourceSampler
from recbole_cdr.data.utils import create_source_samplers

# fixing
from utils.dataloader.crossdomain_dataloader import *
from utils import CrossDomainModelType


def get_cross_domain_dataloader(config, phase, domain='target'):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        domain (str): The domain of Evaldataloader. It can only take two values: 'source' or 'target'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type == CrossDomainModelType.GENERAL:
            return CrossDomainDataloader
    else:
        if domain == 'source':
            return CrossDomainFullSortEvalDataLoader
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def data_preparation_cdr(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (CDRConfig): An instance object of Config, used to record parameter information.
        dataset (CrossDomainDataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        built_datasets = dataset.build()

        source_train_dataset, source_valid_dataset, target_train_dataset, \
            target_valid_dataset, target_test_dataset = built_datasets

        target_train_sampler, target_valid_sampler, target_test_sampler = \
            create_samplers(config, dataset.target_domain_dataset, built_datasets[2:])

        if source_valid_dataset is not None:
            source_train_sampler, source_valid_sampler = create_source_samplers(config, dataset, built_datasets[:2])
            source_valid_data = get_cross_domain_dataloader(config, 'evaluation', 'source')(config, dataset, source_valid_dataset, source_valid_sampler, shuffle=False)
            target_valid_data = get_cross_domain_dataloader(config, 'evaluation', 'target')(config, target_valid_dataset, target_valid_sampler, shuffle=False)

            valid_data = (source_valid_data, target_valid_data)
        else:
            source_train_sampler = CrossDomainSourceSampler(['train'], dataset, config['train_neg_sample_args']['distribution']).set_phase('train')
            valid_data = get_cross_domain_dataloader(config, 'evaluation', 'target')(config, target_valid_dataset, target_valid_sampler, shuffle=False)

        train_data = get_cross_domain_dataloader(config, 'train', 'target')(config, dataset, source_train_dataset, source_train_sampler,
                                                                            target_train_dataset, target_train_sampler, shuffle=True)

        test_data = get_cross_domain_dataloader(config, 'evaluation', 'target')(config, target_test_dataset, target_test_sampler, shuffle=False)

        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, valid_data, test_data