import sys
sys.path.append('../')
import pt_pack as pt
from c_gcns import graph_vqa
import argparse
from torch.optim.lr_scheduler import MultiStepLR


init_kwargs = {
    'model_cls': 'graph_vqa_model',
    'dataset_cls': 'cgs_vqa2_dataset',

    'criterion_cls': 'vqa2_cross_entropy',
    'seed': 1000,
    'verbose': False,

    # 'checkpoint_only_best': False,

    'optimizer_cls': 'adam',
    'optimizer_lr': 1e-4,
    # 'optimizer_final_lr': 0.1,

    'dataset_batch_size': 64,
    'dataset_workers_num': 0,
    'dataset_splits': ('train', 'val'),
    'dataset_shuffles': (True, False),
    'dataset_for_whats': ('train', 'eval'),
    'cuda_is_prefetch': True,

    'logger_cls': 'visdom_logger',
    'logger_splits': ('acc', 'loss'),

    'q_net_hid_dim': 1024,

    'graph_vqa_net_filter_method': 'tri',


    'graph_vqa_net_layer_names': ('cond_graph_conv_layer', 'cond_graph_cls_layer'),
    'graph_vqa_net_layer_node_dims': (2052,) + (1024,),
    'graph_vqa_net_layer_cond_dims': (1024, )*2,
    'graph_vqa_net_layer_edge_dims': (512, )*3,
    'graph_vqa_net_layer_out_dims': (1024,) + (3001, ),
    'graph_vqa_net_layer_methods': ('sum^film-softmax^4^16_none-linear-film^sum_max', 'last'),
    # 'graph_vqa_net_layer_methods': ('cgs:softmax^16_cgs:sum^8_max', 'last'),
    # 'graph_vqa_net_layer_methods': ('cgs:softmax^15_cond:sum^film-linear-film^sum_max', 'last'),
    # 'graph_vqa_net_layer_methods': ('cond:sum^film-softmax^4^15_cgs:sum^8_max', 'last'),
    'graph_vqa_net_layer_dropouts': (0.5,) * 11,

    'cuda_device_ids': (0,)

}


@pt.Trainer.register_hook('INIT', 'append')
def trainer_init(trainer: pt.Trainer):
    scheduler = MultiStepLR(trainer.optimizer.optimizers['graph_vqa_model'], milestones=[30], gamma=0.5)
    scheduler.last_epoch = trainer.last_epoch
    trainer.scheduler = scheduler


@pt.Trainer.register_hook('BEFORE_TRAIN', 'append')
def before_train(trainer: pt.Trainer):
    trainer.scheduler.step()


if __name__ == '__main__':
    trainer = pt.Trainer.build(**init_kwargs)
    trainer()










































