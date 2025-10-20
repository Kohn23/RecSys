from logging import getLogger

from recbole.utils import set_color
from recbole_cdr.data.dataset import CrossDomainSingleDataset, CrossDomainDataset, CrossDomainOverlapDataset


class CrossDomainSingleDatasetFixing(CrossDomainSingleDataset):
    """
        This is a fixing for its parent class
    """

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        self.dataset_path = self.config['data_path']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2token_id = {}
        # fixing
        self.field2bucketnum = {}

        self.field2seqlen = self.config['seq_len'] or {}
        self.alias = {}
        self._preloaded_weight = {}
        self.benchmark_filename_list = self.config['benchmark_filename']
        self.neg_prefix = self.config['NEG_PREFIX']


class CrossDomainDatasetFixing(CrossDomainDataset):
    """
        This is a fixing for its parent class
    """
    def __init__(self, config):
        assert 'source_domain' in config and 'target_domain' in config
        self.config = config
        self.logger = getLogger()
        self.train_modes = config['train_modes']
        self.logger.debug(set_color('Source Domain', 'blue'))
        source_config = config.update(config['source_domain'])
        # fixing
        self.source_domain_dataset = CrossDomainSingleDatasetFixing(source_config, domain='source')

        self.logger.debug(set_color('Target Domain', 'red'))
        target_config = config.update(config['target_domain'])
        # fixing
        self.target_domain_dataset = CrossDomainSingleDatasetFixing(target_config, domain='target')

        self.user_link_dict = None
        self.item_link_dict = None
        self._load_data(config['user_link_file_path'], config['item_link_file_path'])

        # token link remap
        self.source_domain_dataset.remap_user_item_id(self.user_link_dict, self.item_link_dict)

        # user and item ID remap
        self.source_user_ID_remap_dict, self.source_item_ID_remap_dict, \
        self.target_user_ID_remap_dict, self.target_item_ID_remap_dict = self.calculate_user_item_from_both_domain()
        self.source_domain_dataset.remap_user_item_id(self.source_user_ID_remap_dict, self.source_item_ID_remap_dict)
        self.target_domain_dataset.remap_user_item_id(self.target_user_ID_remap_dict, self.target_item_ID_remap_dict)

        # other fields remap
        self.source_domain_dataset.remap_others_id()
        self.target_domain_dataset.remap_others_id()

        # other data process
        self.source_domain_dataset.data_process_after_remap()
        self.target_domain_dataset.data_process_after_remap()
        if self.num_overlap_user > 1:
            self.overlap_dataset = CrossDomainOverlapDataset(config, self.num_overlap_user)
        else:
            self.overlap_dataset = CrossDomainOverlapDataset(config, self.num_overlap_item)
        self.overlap_id_field = self.overlap_dataset.overlap_id_field