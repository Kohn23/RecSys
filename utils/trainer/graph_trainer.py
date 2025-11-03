"""
    Trainers for GNN-based models
"""
import math
from recbole.trainer import Trainer


class HMLETTrainer(Trainer):
    def __init__(self, config, model):
        super(HMLETTrainer, self).__init__(config, model)

        self.warm_up_epochs = config['warm_up_epochs']
        self.ori_temp = config['ori_temp']
        self.min_temp = config['min_temp']
        self.gum_temp_decay = config['gum_temp_decay']
        self.epoch_temp_decay = config['epoch_temp_decay']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx > self.warm_up_epochs:
            # Temp decay
            gum_temp = self.ori_temp * math.exp(-self.gum_temp_decay*(epoch_idx - self.warm_up_epochs))
            self.model.gum_temp = max(gum_temp, self.min_temp)
            self.logger.info(f'Current gumbel softmax temperature: {self.model.gum_temp}')

            for gating in self.model.gating_nets:
                self.model._gating_freeze(gating, True)
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)


class SEPTTrainer(Trainer):
    def __init__(self, config, model):
        super(SEPTTrainer, self).__init__(config, model)
        self.warm_up_epochs = config['warm_up_epochs']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if epoch_idx < self.warm_up_epochs:
            loss_func = self.model.calculate_rec_loss
        else:
            self.model.subgraph_construction()
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)