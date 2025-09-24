"""
    This is the config file for different types of recommender systems

    Note:
            You can also configurate through .yaml files.
            Also, checkout official .yaml config files under property/ for details
"""

# sequential recommender config
config_sr = {
        # General
        'gpu_id': 0,
        'use_gpu': True,
        'seed': 2020,
        'state': 'INFO',
        'reproducibility': True,
        'show_progress': True,
        'save_dataset': False,
        'save_dataloaders': False,
        'benchmark_filename': None,

        # Training
        'checkpoint_dir': 'saved',
        'epochs': 300,
        'train_batch_size': 256,
        'learner': 'adam',
        'learning_rate': 0.0001,
        'eval_step': 1,
        'stopping_step': 5,
        'clip_grad_norm': None,
        'weight_decay': 5,
        'loss_decimal_place': 4,

        # Evaluation
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

        # Dataset
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

        # Models
        # General
        'initializer_range': 0.02,
        'loss_type': 'CE',
        'loss_temp': 0.05,
        'MODEL_TYPE': 'ModelType.SEQUENTIAL',
        'MODEL_INPUT_TYPE': 'InputType.POINTWISE',
        'eval_type': 'EvaluatorType.RANKING',

        # TransformerEncoder
        'n_layers': 1,
        'n_heads': 8,
        'hidden_size': 256,  # same with embedding
        'inner_size': 256, # feedforward size
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,

        # Doc2Vec
        'doc2vec_vector_size': 100,
        'doc2vec_window': 5,
        'doc2vec_min_count': 1,
        'doc2vec_epochs': 10,
        'doc2vec_dm': 1,

        # DSER
        'embedding_size': 8,
        'mlp_layers': [64, 32, 16],
        'dropout_prob': 0,


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

# cross-domain recommender config
config_cdr = {
        # General
        "gpu_id": 0,
        "use_gpu": True,
        "seed": 2022,
        "state": "INFO",
        "reproducibility": True,
        "data_path": "dataset/",
        "checkpoint_dir": "saved",
        "show_progress": True,
        "save_dataset": False,
        "dataset_save_path": None,
        "save_dataloaders": False,
        "dataloaders_save_path": None,
        "log_wandb": False,
        "wandb_project": "recbole_cdr",

        # Training
        "train_epochs": ["BOTH:300"],
        "train_batch_size": 2048,
        "learner": "adam",
        "learning_rate": 0.001,
        "neg_sampling": {
            "uniform": 1,
        },
        "eval_step": 1,
        "stopping_step": 10,
        "clip_grad_norm": None,
        # "clip_grad_norm": {"max_norm": 5, "norm_type": 2},  # if necessary
        "weight_decay": 0.0,
        "loss_decimal_place": 4,
        "require_pow": False,

        # Evaluation
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "split_valid": {"RS": [0.8, 0.2]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
        "repeatable": False,
        "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
        "topk": [10],
        "valid_metric": "MRR@10",
        "valid_metric_bigger": True,
        "eval_batch_size": 4096,
        "metric_decimal_place": 4,

        # Dataset
        "field_separator": "\t",
        "source_domain": {
            "dataset": "ml-1m",
            "data_path": "dataset/",
            "seq_separator": " ",
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating",
            "TIME_FIELD": "timestamp",
            "NEG_PREFIX": "neg_",
            "LABEL_FIELD": "label",
            "load_col": {
                "inter": ["user_id", "item_id", "timestamp"],
            },
            "user_inter_num_interval": "[5,inf)",
            "item_inter_num_interval": "[5,inf)",
            "val_interval": {
                "rating": "[3,inf)",
            },
            "drop_filter_field": True,
        },
        "target_domain": {
            "dataset": "ml-100k",
            "data_path": "dataset/",
            "seq_separator": ",",
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating",
            "TIME_FIELD": "timestamp",
            "NEG_PREFIX": "neg_",
            "LABEL_FIELD": "label",
            "load_col": {
                "inter": ["user_id", "item_id", "timestamp"],
            },
            "user_inter_num_interval": "[5,inf)",
            "item_inter_num_interval": "[5,inf)",
            "val_interval": {
                "rating": "[3,inf)",
            },
            "drop_filter_field": True,
        },
}
