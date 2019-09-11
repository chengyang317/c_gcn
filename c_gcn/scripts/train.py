# coding=utf-8
import sys
sys.path.append('../')
import pt_pack as pt
from c_gcn import graph_vqa


init_params = {
    # env params
    'work_dir': './work_dir',
    'proj_name': 'vqa2_cp_9.11',
    'exp_name': 'lin_con_con_cls',
    'exp_version': 'exp',

    # trainer params
    'trainer_step2prog': 50,


    # optimizer params
    'optimizer_type': 'Adam',
    'optimizer_lr': 3e-4,

    # dataset params
    'dataset_cls': 'graph_vqa2_cp_dataset',
    # 'dataset_req_field_names': ('a_label_scores', 'q_labels'),
    'dataset_batch_size': 64,
    'dataset_splits': ('train', 'test'),
    'dataset_shuffles': (True, False),
    'dataset_belongs': ('train', 'eval'),
    'dataset_num_workers': 0,

    # criterion params
    'criterion_cls': 'vqa2_cross_entropy',

    # cuda params:
    'cuda_cls': 'cuda',
    'cuda_is_prefetch': True,


    # model params
    'model_cls': 'graph_vqa_model',
    'model_q_net_cls': 'cgs_graph_q_net',
    'graph_q_net_dropout': 0.2,
    # 'graph_q_net_module_num': 2,
    # 'graph_vqa_net_filter_method': 'full',
    'graph_vqa_net_filter_method': 'not_eye',
    'graph_vqa_net_layer_names': ('graph_linear_layer',
                                  'cond_graph_conv_layer',
                                  'cond_graph_conv_layer',
                                  # 'cond_graph_conv_layer',
                                  # 'cond_graph_conv_layer',
                                  'cond_graph_cls_layer'
                                  ),
    'graph_vqa_net_layer_node_dims': (2052,) + (1028,) * 2 + (1024*4,),
    'graph_vqa_net_layer_cond_dims': (1024, )*11,
    'graph_vqa_net_layer_edge_dims': (512, )*11,
    'graph_vqa_net_layer_out_dims': (1024,) * 3 + (3001, ),
    'graph_vqa_net_layer_params': ('linear',
                                    '{'
                                    '"edge": {"feat": "mul^film", "weight": "softmax^1^8"},'
                                    '"conv": {"feat": "none", "param": "linear", "node": "film^sum", "weight": "node^linear^sigmoid"},'
                                    '"pool": "weight^mix"'
                                    '}',
                                    '{'
                                    '"edge": {"feat": "share", "weight": "softmax^share^8"},'
                                    '"conv": {"feat": "none", "param": "share", "node": "film^sum", "weight": "node^share^sigmoid"},'
                                    '"pool": "weight^mix"'
                                    '}',
                                    # '{'
                                    # '"edge": {"feat": "cat^film", "weight": "softmax^1^8"},'
                                    # '"conv": {"feat": "none", "param": "linear", "node": "film^sum", "weight": "node^linear"},'
                                    # '"pool": "weight^max"'
                                    # '}',
                                    # '{'
                                    # '"edge": {"feat": "share", "weight": "softmax^share^8"},'
                                    # '"conv": {"feat": "none", "param": "share", "node": "film^sum", "weight": "node^share"},'
                                    # '"pool": "weight^mix"'
                                    # '}',
                                    'linear_cat'),
    'graph_vqa_net_layer_dropouts': (0.2,) * 11,

}


controller = pt.Controller(init_params)
controller.train()




