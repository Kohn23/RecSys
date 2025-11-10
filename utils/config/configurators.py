import os
from enum import Enum
from recbole.evaluator import metric_types, smaller_metrics
from recbole.utils import EvaluatorType, ModelType, InputType
from recbole.config import Config

from utils import train_mode2state, CrossDomainModelType


# TODO：Fix
# class CDRConfigFixing(CDRConfig):
#     """
#     This is a fixing for CDRConfig
#     """
#
#     def __init__(self, model=None, config_file_list=None, config_dict=None):
#         """
#         Args:
#             model (str/CrossDomainRecommender): the model name or the model class, default is None, if it is None, config
#             will search the parameter 'model' from the external input as the model name or model class.
#             from the external input as the dataset name.
#             config_file_list (list of str): the external config file, it allows multiple config files, default is None.
#             config_dict (dict): the external parameter dictionaries, default is None.
#         """
#         self.compatibility_settings()
#         self._init_parameters_category()
#         self.parameters['Dataset'] += ['source_domain', 'target_domain']
#         self.yaml_loader = self._build_yaml_loader()
#         self.file_config_dict = self._remove_domain_prefix(
#             self._load_config_files(config_file_list))
#         self.variable_config_dict = self._remove_domain_prefix(
#             self._load_variable_config_dict(config_dict))
#         self.cmd_config_dict = self._remove_domain_prefix(self._load_cmd_line())
#         self._merge_external_config_dict()
#
#         self.model, self.model_class, self.dataset = self._get_model_and_dataset(model)
#         self._load_internal_config_dict(self.model, self.model_class, self.dataset)
#         self.final_config_dict = self._get_final_config_dict()
#
#         # debug
#         # print(self.final_config_dict['eval_args'])
#
#         self._set_default_parameters()
#         self._init_device()
#         self._set_train_neg_sample_args()
#         # fixing
#         self._set_eval_neg_sample_args("valid")
#         self._set_eval_neg_sample_args("test")
#
#         self.dataset = self._check_cross_domain()
#
#     def _load_internal_config_dict(self, model, model_class, dataset):
#         current_path = os.path.dirname(os.path.realpath(__file__))
#         overall_init_file = os.path.join(current_path, 'properties/overall.yaml')
#         model_init_file = os.path.join(current_path, 'properties/model/' + model + '.yaml')
#         sample_init_file = os.path.join(current_path, 'properties/dataset/cross_domain.yaml')
#
#         self.internal_config_dict = dict()
#         for file in [overall_init_file, model_init_file, sample_init_file]:
#             if os.path.isfile(file):
#                 self._update_internal_config_dict(file)
#
#         self.internal_config_dict['MODEL_TYPE'] = model_class.type
#
#     def _set_default_parameters(self):
#         self.final_config_dict['model'] = self.model
#
#         if hasattr(self.model_class, 'input_type'):
#             self.final_config_dict['MODEL_INPUT_TYPE'] = self.model_class.input_type
#         elif 'loss_type' in self.final_config_dict:
#             if self.final_config_dict['loss_type'] in ['CE']:
#                 if self.final_config_dict['MODEL_TYPE'] == ModelType.SEQUENTIAL and \
#                         self.final_config_dict['neg_sampling'] is not None:
#                     raise ValueError(f"neg_sampling [{self.final_config_dict['neg_sampling']}] should be None "
#                                      f"when the loss_type is CE.")
#                 self.final_config_dict['MODEL_INPUT_TYPE'] = InputType.POINTWISE
#             elif self.final_config_dict['loss_type'] in ['BPR']:
#                 self.final_config_dict['MODEL_INPUT_TYPE'] = InputType.PAIRWISE
#         else:
#             raise ValueError('Either Model has attr \'input_type\',' 'or arg \'loss_type\' should exist in config.')
#
#         metrics = self.final_config_dict['metrics']
#         if isinstance(metrics, str):
#             self.final_config_dict['metrics'] = [metrics]
#
#         eval_type = set()
#         for metric in self.final_config_dict['metrics']:
#             if metric.lower() in metric_types:
#                 eval_type.add(metric_types[metric.lower()])
#             else:
#                 raise NotImplementedError(f"There is no metric named '{metric}'")
#         if len(eval_type) > 1:
#             raise RuntimeError('Ranking metrics and value metrics can not be used at the same time.')
#         self.final_config_dict['eval_type'] = eval_type.pop()
#
#         if self.final_config_dict['MODEL_TYPE'] == ModelType.SEQUENTIAL and not self.final_config_dict['repeatable']:
#             raise ValueError('Sequential models currently only support repeatable recommendation, '
#                              'please set `repeatable` as `True`.')
#
#         valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
#         self.final_config_dict['valid_metric_bigger'] = False if valid_metric.lower() in smaller_metrics else True
#
#         topk = self.final_config_dict['topk']
#         if isinstance(topk, (int, list)):
#             if isinstance(topk, int):
#                 topk = [topk]
#             for k in topk:
#                 if k <= 0:
#                     raise ValueError(
#                         f'topk must be a positive integer or a list of positive integers, but get `{k}`'
#                     )
#             self.final_config_dict['topk'] = topk
#         else:
#             raise TypeError(f'The topk [{topk}] must be a integer, list')
#
#         if 'additional_feat_suffix' in self.final_config_dict:
#             ad_suf = self.final_config_dict['additional_feat_suffix']
#             if isinstance(ad_suf, str):
#                 self.final_config_dict['additional_feat_suffix'] = [ad_suf]
#
#         # eval_args checking
#         default_eval_args = {
#             'split': {'RS': [0.8, 0.1, 0.1]},
#             'order': 'RO',
#             'group_by': 'user',
#
#             # fixing
#             'mode': {"valid": "full", "test": "full"},
#         }
#         if not isinstance(self.final_config_dict['eval_args'], dict):
#             raise ValueError(f"eval_args:[{self.final_config_dict['eval_args']}] should be a dict.")
#
#         for op_args in default_eval_args:
#             if op_args not in self.final_config_dict['eval_args']:
#                 self.final_config_dict['eval_args'][op_args] = default_eval_args[op_args]
#
#         if isinstance(self.final_config_dict['eval_args']['mode'], str):
#             mode = self.final_config_dict['eval_args']['mode']
#             self.final_config_dict['eval_args']['mode'] = {'valid':mode, 'test':mode}
#
#         if (self.final_config_dict['eval_args']['mode'] == 'full'
#                 and self.final_config_dict['eval_type'] == EvaluatorType.VALUE):
#             raise NotImplementedError('Full sort evaluation do not match value-based metrics!')
#
#         # training_mode args
#         train_scheme = []
#         train_epochs = []
#         for train_arg in self.final_config_dict['train_epochs']:
#             scheme, epoch = train_arg.split(':')
#             if scheme not in train_mode2state:
#                 raise ValueError(f"[{scheme}] is not a supported training mode.")
#             train_scheme.append(scheme)
#             train_epochs.append(epoch)
#         self.final_config_dict['train_modes'] = train_scheme
#         self.final_config_dict['epoch_num'] = train_epochs
#         source_split_flag = True if 'SOURCE' in train_scheme else False
#         self.final_config_dict['source_split'] = source_split_flag
#         self.final_config_dict['epochs'] = int(train_epochs[0])
#
#     def _convert_config_dict(self, config_dict):
#         """
#         This function convert the str parameters to their original type.
#         """
#         for key in config_dict:
#             param = config_dict[key]
#             if not isinstance(param, str):
#                 continue
#
#             # fixing
#             if key == "MODEL_TYPE":
#                 try:
#                     param = param.split('.')
#                     if param[0] != "CrossDomainModelType":
#                         raise ValueError(f"wrong prefix:{param[0]}")
#                     config_dict[key] = CrossDomainModelType[param[1].upper()]
#                     continue
#                 except KeyError:
#                     raise ValueError(f"Unsupported model type: {param}")
#
#             try:
#                 value = eval(param)
#                 if value is not None and not isinstance(
#                         value, (str, int, float, list, tuple, dict, bool, Enum)
#                 ):
#                     value = param
#             except (NameError, SyntaxError, TypeError):
#                 if isinstance(param, str):
#                     if param.lower() == "true":
#                         value = True
#                     elif param.lower() == "false":
#                         value = False
#                     else:
#                         value = param
#                 else:
#                     value = param
#             config_dict[key] = value
#         return config_dict
