# coding=utf-8
from pt_pack.criterions.base import Criterion
import torch.nn as nn
import torch


__all__ = ['Vqa2CrossEntropy', 'ClevrCrossEntropy', 'Vqa2NewCrossEntropy']


class Vqa2CrossEntropy(Criterion):
    loss_cls = nn.MultiLabelSoftMarginLoss

    def __init__(self):
        super().__init__()

    def forward(self, message):
        logits = message['logits']['value']
        batch_data = message['batch_data']['value']
        answer = batch_data['a_label_scores']
        a_votes = batch_data['a_label_counts']
        loss = self.loss_l(logits, answer)
        correct = self.compute_score(logits.data, a_votes)
        b_size = logits.size(0)
        acc = correct / b_size
        log = {
            'loss': {'name': 'loss', 'value': loss, 'tags': ('keep', 'tf', 'prog', 'mean')},
            'acc': {'name': 'acc', 'value': acc, 'tags': ('keep', 'tf', 'prog', 'mean')},
        }
        message.update(log)
        return message

    @staticmethod
    def compute_score(logits, a_votes):
        pred = logits.max(1)[1]
        # answer = a_votes > 0
        answer = torch.min(a_votes/3, torch.tensor(1.0).cuda(pred.device))
        score = answer.gather(dim=-1, index=pred.unsqueeze(-1)).squeeze()
        return score.sum().item()

    def evaluate(self, model_out: torch.Tensor, sample):
        q_ids = sample['q_ids']
        a_votes = sample['a_label_counts']
        correct = self.compute_score(model_out.data, a_votes)
        b_size = model_out.size(0)
        acc = correct / b_size
        a_ids = model_out.max(1)[1]
        log = {'acc': acc, 'correct': correct.tolist(), 'q_ids': q_ids.tolist(), 'loss': None, 'a_ids': a_ids.tolist()}
        return log

    def test(self, model_out: torch.Tensor, sample):
        q_ids = sample['q_ids']
        a_ids = model_out.max(1)[1]
        log = {'q_ids': q_ids.tolist(), 'a_ids': a_ids.tolist(), 'acc': 0., 'correct': 0., 'loss': 0.}
        return log


class Vqa2NewCrossEntropy(Criterion):
    loss_cls = nn.BCEWithLogitsLoss

    def __init__(self,
                 logger_group=None
                 ):
        super().__init__()
        self.logger_group = logger_group or LoggerGroup.logger_group
        self.loss_logger = self.logger_group.register_logger('loss')
        self.acc_logger = self.logger_group.register_logger('acc')

    def forward(self, model_out: torch.Tensor, sample):
        a_scores = sample['a_label_scores']
        # a_votes = sample['a_label_counts']
        loss = self.loss_l(model_out, a_scores)
        correct = self.compute_score(model_out.data, a_scores)
        b_size = model_out.size(0)
        acc = correct / b_size
        # log = {'loss': loss.item(), 'acc': acc, 'correct': correct.cpu().numpy()}
        log = {'loss': loss.item(), 'acc': acc}
        return loss, log

    @staticmethod
    def compute_score(logits, a_scores):
        pred = logits.max(1)[1]
        score = a_scores.gather(dim=-1, index=pred.unsqueeze(-1)).squeeze()
        return score.sum().item()

    def evaluate(self, model_out: torch.Tensor, sample):
        q_ids = sample['q_ids']
        a_votes = sample['a_label_counts']
        correct = self.compute_score(model_out.data, a_votes)
        b_size = model_out.size(0)
        acc = correct / b_size
        a_ids = model_out.max(1)[1]
        log = {'acc': acc, 'correct': correct.tolist(), 'q_ids': q_ids.tolist(), 'loss': None, 'a_ids': a_ids.tolist()}
        return log

    def test(self, model_out: torch.Tensor, sample):
        q_ids = sample['q_ids']
        a_ids = model_out.max(1)[1]
        log = {'q_ids': q_ids.tolist(), 'a_ids': a_ids.tolist(), 'acc': 0., 'correct': 0., 'loss': 0.}
        return log



class ClevrCrossEntropy(Criterion):
    loss_cls = nn.CrossEntropyLoss

    def __init__(self,
                 logger_group=None
                 ):
        super().__init__()
        self.logger_group = logger_group or LoggerGroup.logger_group
        self.loss_logger = self.logger_group.register_logger('loss')
        self.acc_logger = self.logger_group.register_logger('acc')

    def forward(self, model_out: torch.Tensor, sample):
        answer = sample['a_labels']
        loss = self.loss_l(model_out, answer)
        correct = model_out.data.max(1)[1] == answer
        acc = correct.sum().item() / model_out.size(0)
        log = {'loss': loss.item(), 'acc': acc}
        return loss, log

    def evaluate(self, model_out: torch.Tensor, sample):
        answer = sample['a_labels']
        q_ids = sample['q_ids']
        correct = model_out.data.max(1)[1] == answer
        acc = correct.sum().item() / model_out.size(0)
        log = {'acc': acc, 'correct': correct.tolist(), 'q_ids': q_ids.tolist(), 'loss': None}
        return log









