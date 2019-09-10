# coding=utf-8
import pt_pack as pt
import torch.nn as nn
from tqdm import tqdm
import torch


init_kwargs = {
    'model_cls': 'graph_vqa_model',
    'dataset_cls': 'graph_vqa2_dataset',

    'criterion_cls': 'vqa2_cross_entropy',
    'seed': 1,
    'verbose': False,

    # 'checkpoint_only_best': False,

    'optimizer_cls': 'adam',
    'optimizer_lr': 3e-4,
    # 'optimizer_final_lr': 0.1,

    'dataset_batch_size': 2,
    'dataset_workers_num': 0,
    'dataset_splits': ('train_val', 'val'),
    'cuda_dataset_prefetch': False,

    'logger_logger_cls': 'visdom_logger',

    'q_net_hid_dim': 1024,

    'graph_vqa_net_filter_method': 'tri_u',


    'graph_vqa_net_layers': ('graph_linear_layer', 'cond_graph_conv_layer',
                             'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             'cond_graph_pool_layer', 'cond_graph_conv_layer',
                             'cond_graph_cls_layer'),
    'graph_vqa_net_layer_node_dims': (2052,) + (1028,) * 9 + (1024*5,),
    'graph_vqa_net_layer_cond_dims': (1024, )*11,
    'graph_vqa_net_layer_edge_dims': (512, )*11,
    'graph_vqa_net_layer_out_dims': (1024,) * 10 + (3001, ),
    'graph_vqa_net_layer_methods': ('linear', 'sum^film_softmax^4^12_linear_film^sum',
                                    'share_softmax^share^18', 'share_softmax^share^6_share_film^sum',
                                    'share_softmax^share^9', 'share_softmax^share^4_share_film^sum',
                                    'share_softmax^share^4', 'share_softmax^share^2_share_film^sum',
                                    'share_softmax^share^2', 'share_softmax^share^1_share_film^sum',
                                    'linear_cat'),
    'graph_vqa_net_layer_dropouts': (0.2,) * 11,


}


def epoch_eval_fn(trainer,
                  epoch_idx,
                  model=None,
                  loader=None,
                  criterion=None,
                  logger_group=None,
                  cuda=None,
                  optimizer=None,
                  checkpoint=None
                  ):
    if not trainer.need_val(epoch_idx):
        return
    epoch_idx = epoch_idx
    model: pt.Model = model or trainer.model
    if isinstance(model, nn.DataParallel):
        model.module.set_mode(False)
    else:
        model.set_mode(False)
    loader = loader or trainer.loaders['val']
    criterion: pt.Criterion = criterion or trainer.criterion
    logger_group: pt.LoggerGroup = logger_group or trainer.logger_group
    cuda: pt.Cuda = cuda or trainer.cuda
    pbar = tqdm(range(len(loader)))
    loader = cuda.process_loader(loader)

    a_vocab = loader.dataset.answer_vocab
    ret_json = list()
    for step_idx in pbar:
        sample = loader.next()
        with torch.no_grad():
            model_input = pt.get_model_input(model, sample)
            if isinstance(model_input, dict):
                model_output = model(**model_input)
            else:
                model_output = model(*model_input)
            log = criterion.evaluate(model_output, sample)
        logger_group.forward(log, step_idx, epoch_idx, is_train=False, need_log=False)

        for q_id, a_id in zip(log['q_ids'], log['a_ids']):
            ret_json.append({'question_id': int(q_id), 'answer': a_vocab[a_id]})
    pbar.close()
    logger_group.epoch_update(epoch_idx, is_train=False)


trainer = pt.Trainer.build(**init_kwargs)
trainer()



