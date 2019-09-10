# coding=utf-8
import sys
sys.path.append('../')

import pt_pack as pt
from c_gcns import graph_vqa


init_kwargs = {
    'model_cls': 'graph_vqa_model',
    'model_graph_net_cls': 'res_graph_vqa_net',
    'dataset_cls': 'graph_vqa2_cp_dataset',

    'criterion_cls': 'vqa2_cross_entropy',
    'seed': 1,
    'verbose': False,

    # 'checkpoint_only_best': False,

    'optimizer_cls': 'adam',
    'optimizer_lr': 3e-4,
    # 'optimizer_final_lr': 0.1,

    'dataset_batch_size': 55,
    'dataset_splits': ('train', 'test'),
    'dataset_shuffles': (True, False),
    'dataset_for_whats': ('train', 'eval'),
    'dataset_workers_num': 0,
    'cuda_is_prefetch': True,

    'logger_cls': 'visdom_logger',
    'logger_splits': ('loss', 'acc'),

    'graph_vqa_net_filter_method': 'full',
    # 'graph_vqa_net_filter_method': 'not_eye',


    'graph_vqa_net_layer_names': ('graph_linear_layer', 'cond_graph_conv_layer',
                             # 'cond_graph_conv_layer', 'cond_graph_conv_layer',
                             'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             # 'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             'cond_graph_cls_layer'),
    'graph_vqa_net_layer_node_dims': (2052,) + (1028,) * 5 + (1024*1,),
    'graph_vqa_net_layer_cond_dims': (1024, )*7,
    'graph_vqa_net_layer_edge_dims': (512, )*7,
    'graph_vqa_net_layer_out_dims': (1024,) * 6 + (3001, ),
    'graph_vqa_net_layer_methods': ('linear', 'cat^film-softmax^4^8_none-linear-film^sum_max',
                                    # 'share-softmax^share^8_none-share-film^sum_max', 'share-softmax^share^8_none-share-film^sum_max',
                                    'share_softmax^share^18', 'share-softmax^share^6_none-share-film^sum_max',
                                    'share_softmax^share^9', 'share-softmax^share^4_none-share-film^sum_max',
                                    # 'share_softmax^share^4', 'share_softmax^share^2_share_film^sum',
                                    # 'share_softmax^share^2', 'share_softmax^share^1_share_film^sum',
                                    'linear_last'),
    'graph_vqa_net_layer_dropouts': (0.2,) * 11,
    'cuda_device_ids': (0,)
}


trainer = pt.Trainer.build(**init_kwargs)
trainer()



