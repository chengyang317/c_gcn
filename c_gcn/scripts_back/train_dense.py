# coding=utf-8
import sys
sys.path.append('../')

import pt_pack as pt
from c_gcns import graph_vqa


init_kwargs = {
    'model_cls': 'graph_vqa_model',
    'dataset_cls': 'graph_vqa2_cp_dataset',

    'criterion_cls': 'vqa2_cross_entropy',
    'seed': 1,
    'verbose': False,

    # 'checkpoint_only_best': False,

    'optimizer_cls': 'adam',
    'optimizer_lr': 3e-4,
    # 'optimizer_final_lr': 0.1,

    'dataset_batch_size': 64,
    'dataset_splits': ('train', 'test'),
    'dataset_shuffles': (True, False),
    'dataset_for_whats': ('train', 'eval'),
    'dataset_workers_num': 0,
    'cuda_is_prefetch': True,

    'logger_cls': 'visdom_logger',
    'logger_splits': ('loss', 'acc'),

    # 'graph_vqa_net_filter_method': 'full',
    'graph_vqa_net_filter_method': 'not_eye',


    'graph_vqa_net_layer_names': ('graph_linear_layer', 'cond_graph_conv_layer',
                             'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             # 'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             # 'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             # 'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             'cond_graph_cls_layer'),
    'graph_vqa_net_layer_node_dims': (2052,) + (1028,) * 4 + (1024*4,),
    'graph_vqa_net_layer_cond_dims': (1024, )*11,
    'graph_vqa_net_layer_edge_dims': (512, )*11,
    'graph_vqa_net_layer_out_dims': (1024,) * 5 + (3001, ),
    'graph_vqa_net_layer_methods': ('linear', 'cat^film-softmax^1^8_none-linear-film^sum-node^linear_mix',
                                    'node^linear_18', 'share-softmax^share^6_none-share-film^sum_max',
                                    # 'none-softmax^node^9', 'share-softmax^share^4_none-share-film^sum_max',
                                    # 'none-softmax^node^4', 'share-softmax^share^2_none-share-film^sum_max',
                                    # 'none-softmax^inherit^2', 'share-softmax^share^1_none-share-film^sum_max',
                                    'linear_cat'),
    'graph_vqa_net_layer_dropouts': (0.2,) * 11,

}


trainer = pt.Trainer.build(**init_kwargs)
trainer()



