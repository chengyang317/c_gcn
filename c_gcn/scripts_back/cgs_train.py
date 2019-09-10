# coding=utf-8
import sys
sys.path.append('../')
import pt_pack as pt
from c_gcns import graph_vqa


init_kwargs = {
    'model_cls': 'graph_vqa_model',
    'dataset_cls': 'cgs_vqa2_dataset',

    'criterion_cls': 'vqa2_cross_entropy',
    'seed': 1000,
    'verbose': False,

    # 'checkpoint_only_best': False,

    'optimizer_cls': 'adam',
    'optimizer_lr': 3e-4,
    # 'optimizer_final_lr': 0.1,

    'dataset_batch_size': 64,
    'dataset_workers_num': 2,
    'dataset_splits': ('train', 'val'),
    'dataset_shuffles': (True, False),
    'dataset_for_whats': ('train', 'eval'),
    'cuda_is_prefetch': False,

    'logger_cls': 'visdom_logger',
    'logger_splits': ('acc', 'loss'),

    'q_net_hid_dim': 1024,

    'graph_vqa_net_filter_method': 'tri',


    'graph_vqa_net_layer_names': ('graph_conv_layer', 'cgs_graph_cls_layer'),
    'graph_vqa_net_layer_node_dims': (2052,) + (1024,),
    'graph_vqa_net_layer_cond_dims': (1024, )*2,
    'graph_vqa_net_layer_edge_dims': (512, )*3,
    'graph_vqa_net_layer_out_dims': (1024,) + (3001, ),
    # 'graph_vqa_net_layer_methods': ('max^film_softmax^4^12_linear_film^sum', 'last'),
    'graph_vqa_net_layer_methods': ('cgs:softmax^16_cgs:sum^8_max', 'last'),
    # 'graph_vqa_net_layer_methods': ('cgs:softmax^15_cond:sum^film-linear-film^sum_max', 'last'),
    # 'graph_vqa_net_layer_methods': ('cond:sum^film-softmax^4^15_cgs:sum^8_max', 'last'),
    'graph_vqa_net_layer_dropouts': (0.5,) * 11,

    'cuda_device_ids': (5,)

}


trainer = pt.Trainer.build(**init_kwargs)
trainer()



