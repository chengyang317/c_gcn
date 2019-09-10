import pt_pack as pt
import torch.nn as nn
from tqdm import tqdm
import torch


# init_kwargs = {
#     'model_cls': 'graph_vqa_model',
#     'dataset_cls': 'graph_vqa2_dataset',
#
#     'criterion_cls': 'vqa2_cross_entropy',
#     'seed': 1,
#     'verbose': False,
#
#     # 'checkpoint_only_best': False,
#
#     'optimizer_cls': 'adam',
#     'optimizer_lr': 3e-4,
#     # 'optimizer_final_lr': 0.1,
#
#     'dataset_batch_size': 64,
#     'dataset_splits': ('val',),
#     'dataset_workers_num': 0,
#     'cuda_dataset_prefetch': False,
#
#     'logger_logger_cls': 'visdom_logger',
#
#     'q_net_hid_dim': 1024,
#
#     'graph_vqa_net_filter_method': 'tri_u',
#     # 'graph_vqa_net_filter_method': 'not_eye',
#
#
#     'graph_vqa_net_layers': ('graph_linear_layer', 'cond_graph_conv_layer',
#                              'cond_graph_pool_layer', 'cond_graph_conv_layer',
#                              'cond_graph_pool_layer', 'cond_graph_conv_layer',
#                              'cond_graph_pool_layer', 'cond_graph_conv_layer',
#                              'cond_graph_pool_layer', 'cond_graph_conv_layer',
#                              'cond_graph_cls_layer'),
#     'graph_vqa_net_layer_node_dims': (2052,) + (1028,) * 9 + (1024*5,),
#     'graph_vqa_net_layer_cond_dims': (1024, )*11,
#     'graph_vqa_net_layer_edge_dims': (512, )*11,
#     'graph_vqa_net_layer_out_dims': (1024,) * 10 + (3001, ),
#     'graph_vqa_net_layer_methods': ('linear', 'sum^film_softmax^4^12_linear_film^sum',
#                                     'share_softmax^share^18', 'share_softmax^share^6_share_film^sum',
#                                     'share_softmax^share^9', 'share_softmax^share^4_share_film^sum',
#                                     'share_softmax^share^4', 'share_softmax^share^2_share_film^sum',
#                                     'share_softmax^share^2', 'share_softmax^share^1_share_film^sum',
#                                     'linear_cat'),
#     'graph_vqa_net_layer_dropouts': (0.2,) * 11,
#
# }

init_kwargs = {
    'model_cls': 'graph_vqa_model',
    'dataset_cls': 'cgs_vqa2_dataset',
    'dataset_splits': ('val',),
    'cfg_file': 'nips.1layer.CgsDs',
    'cuda_device_ids': 4,
}


def eval_fn(evaluator):
    model = evaluator.model
    if isinstance(model, nn.DataParallel):
        model.module.set_mode(False)
    else:
        model.set_mode(False)
    loader = evaluator.loaders[0]
    pbar = tqdm(range(len(loader)))
    cuda_loader = evaluator.cuda.process_loader(loader)
    scores = list()
    for _ in pbar:
        # if step_idx > 10:
        #     break
        sample = cuda_loader.next()
        with torch.no_grad():
            model_input = pt.get_model_input(model, sample)
            if isinstance(model_input, dict):
                logits = model(**model_input)
            else:
                logits = model(*model_input)
        a_votes = sample['a_label_counts']
        pred = logits.max(1)[1]
        answer_score = torch.min(a_votes / 3, torch.tensor(1.0).cuda(pred.device))
        score = answer_score.gather(dim=-1, index=pred.unsqueeze(-1)).squeeze()
        batch_score = score.sum().item()
        scores.append(batch_score)
    pbar.close()
    acc = sum(scores) / len(loader.dataset)
    print(f'The acc is : {acc} for epoch {evaluator.checkpoint.last_epoch}')


evaluator = pt.Evaluator.build_modules(**init_kwargs)
evaluator(evaluate_fn=eval_fn)


