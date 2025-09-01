# # single-domain
# from recbole.config import Config
# from recbole.data import create_dataset, data_preparation
# from recbole.trainer import Trainer

# cross-domain
from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.trainer import Trainer
from recbole_cdr.quick_start import quick_start

from models import CoNet


config_dict = {
    'source_domain': {
        'dataset': 'ml-100k',
        # 可以添加源域特定的参数
        'min_interactions': 5
    },
    'target_domain': {
        'dataset': 'ml-1m',
        # 可以添加目标域特定的参数
        'min_interactions': 10
    },

    # trainer
    'checkpoint_dir': './bin/checkpoints',
    
    # logger_path isn't manageable
}


config = CDRConfig(model=CoNet, config_dict=config_dict)

# 准备数据
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# 初始化模型
model = CoNet(config, train_data.dataset).to(config['device'])

# 训练与评估
trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
test_result = trainer.evaluate(test_data)

