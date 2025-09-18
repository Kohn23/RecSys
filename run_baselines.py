"""
    You can also run this script on terminal/with config
"""

from recbole.quick_start import run_recbole
from recbole.model.sequential_recommender import BERT4Rec, SASRec
from recbole_cdr.model.cross_domain_recommender import bitgcf
from recbole.trainer.trainer import Trainer


if __name__ == "__main__":

    config_dict = {
        # General Hyper Parameters
        'gpu_id': 0,
        'use_gpu': True,
        'seed': 2020,
        'state': 'INFO',
        'reproducibility': True,
        'show_progress': True,
        'save_dataset': False,
        'save_dataloaders': False,
        'benchmark_filename': None,

        # Training Hyper Parameters
        'checkpoint_dir': 'saved',
        'epochs': 300,
        'train_batch_size': 128,
        'learner': 'adam',
        'learning_rate': 0.0001,
        'eval_step': 1,
        'stopping_step': 10,
        'clip_grad_norm': None,
        'weight_decay': 5.0,
        'loss_decimal_place': 4,

        # Evaluation Hyper Parameters
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'mode': 'uni100',
            'group_by': 'user'
        },
        'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
        'topk': [5, 10],
        'valid_metric': 'MRR@10',
        'valid_metric_bigger': True,
        'eval_batch_size': 1024,
        'metric_decimal_place': 4,

        # Dataset Hyper Parameters
        'field_separator': '\t',
        'seq_separator': ' ',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'seq_len': None,
        'LABEL_FIELD': 'label',
        'threshold': None,
        'NEG_PREFIX': 'neg_',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp']
        },
        'unload_col': None,
        'unused_col': None,
        'additional_feat_suffix': None,
        'rm_dup_inter': None,
        'val_interval': None,
        'filter_inter_by_user_or_item': True,
        'user_inter_num_interval': [0, float('inf')],
        'item_inter_num_interval': [0, float('inf')],
        'alias_of_user_id': None,
        'alias_of_item_id': None,
        'alias_of_entity_id': None,
        'alias_of_relation_id': None,
        'preload_weight': None,
        'normalize_field': None,
        'normalize_all': None,
        'ITEM_LIST_LENGTH_FIELD': 'item_length',
        'LIST_SUFFIX': '_list',
        'MAX_ITEM_LIST_LENGTH': 200,
        'POSITION_FIELD': 'position_id',
        'HEAD_ENTITY_ID_FIELD': 'head_id',
        'TAIL_ENTITY_ID_FIELD': 'tail_id',
        'RELATION_ID_FIELD': 'relation_id',
        'ENTITY_ID_FIELD': 'entity_id',

        # Model Hyper
        'n_layers': 2,
        'n_heads': 1,
        'hidden_size': 1,  # same with embedding
        'inner_size': 256,
        'hidden_dropout_prob': 0.2,
        'attn_dropout_prob': 0.2,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'loss_type': 'CE',
        'MODEL_TYPE': 'ModelType.SEQUENTIAL',
        'dropout_prob': 0.5,
        'MODEL_INPUT_TYPE': 'InputType.POINTWISE',
        'eval_type': 'EvaluatorType.RANKING',

        # Other Hyper Parameters
        'neg_sampling': None,
        'multi_gpus': False,
        'repeatable': True,
        'device': 'cuda',
        # 'train_neg_sample_args': {
        #     'distribution': 'uniform',
        #     'sample_num': 1
        # },
        'train_neg_sample_args': None,
        'eval_neg_sample_args': {
            'strategy': 'by',
            'by': 999,
            'distribution': 'uniform'
        }
    }

    run_recbole(model='FEARec', dataset='amv_movies', config_dict=config_dict)
