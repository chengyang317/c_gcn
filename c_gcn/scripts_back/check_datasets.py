import pt_pack as pt

# for split in ('train', 'test'):
#     train_val_dataset = pt.GraphVqa2Dataset(split=split)
#     for i in range(10):
#         print(train_val_dataset[i])

# dataset = pt.GraphVqa2Dataset(split='train', req_field_names=['q_tokens', 'img_ids'])
#
# for idx, img_id in enumerate(dataset['img_ids']):
#     if int(img_id) == 575:
#         q_tokens = dataset['q_tokens'][idx]
#         print(q_tokens)

# for idx, q_tokens in enumerate(dataset['q_tokens']):
#     for q_token in q_tokens:
#         if q_token == 'horse':
#             print(idx)



for split in ('val', 'train_val', 'test'):
    dataset = pt.CgsVqa2Dataset(split=split)
    print(dataset[0])
