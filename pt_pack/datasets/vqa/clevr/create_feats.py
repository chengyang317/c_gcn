# coding=utf-8
import pt_pack as pt
from torchvision import transforms
import tqdm
import h5py
import torch


def ImgReader(data_dir, split):
    tfs = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.Pad(4),
        transforms.RandomCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if split in ('valB_30k', 'valB_120k'):
        split = 'valB'
    img_reader = pt.ImgDirReader(data_dir.joinpath(f'images/{split}'), transforms=tfs)
    return img_reader


init_kwargs = {
    'cfg_file': 'film*q-1024*init-A-48_dsod-dp2-96-r24_dsod-dp2-96-r24*4dense*cls*adam5e4',
    'cuda_device_ids': (0, ),

    'dataset_cls': 'clevr_dataset',
    'dataset_req_field_names': ('img_ids', 'images'),


    'film_img_net_block_depths': (None, 2, 2),
    'film_img_net_growth_rates': (None, 24, 24),
}


parser = pt.Parser.build(**init_kwargs)
args = parser.args
model = pt.Model.build(args)
checkpoint = pt.Checkpoint.build(args)
cuda = pt.Cuda.build(args)
checkpoint.load_checkpoint(model)
model = cuda.process_model(model)


for split in ('train', 'val'):
    data_dir = pt.to_path(args.dataset_data_dir)
    h5_file = data_dir.joinpath(f'clevr/{split}_dsod_28_96.h5')
    if h5_file.exists():
        continue

    h5_fd = h5py.File(h5_file, 'w')
    reader = ImgReader(args.dataset_data_dir, split)
    dset_len = len(reader['img_ids'])
    batch_size = 64

    batch_img_ids = list()
    batch_images = list()
    d_sets = dict()
    start_id = 0
    for dset_idx in tqdm.tqdm(range(dset_len)):
        batch_img_ids.append(reader['img_ids'][dset_idx])
        batch_images.append(reader['images'][dset_idx])
        if len(batch_images) == batch_size:
            if 'img_ids' not in d_sets:
                d_sets['img_ids'] = h5_fd.create_dataset('img_ids', (dset_len,))
            d_sets['img_ids'][start_id:start_id+batch_size] = pt.to_numpy(batch_img_ids)
            with torch.no_grad():
                feats = model.img_net(cuda.process_sample(pt.to_tensor(batch_images)))
            if 'feats' not in d_sets:
                d_sets['feats'] = h5_fd.create_dataset('feats', (dset_len, *feats.shape[1:]))
            d_sets['feats'][start_id:start_id+batch_size] = pt.to_numpy(feats)
            batch_img_ids, batch_images = list(), list()
            start_id += batch_size
    if len(batch_img_ids) > 0:
        d_sets['img_ids'][start_id:start_id + len(batch_img_ids)] = pt.to_numpy(batch_img_ids)
        with torch.no_grad():
            feats = model.img_net(cuda.process_sample(pt.to_tensor(batch_images)))
        d_sets['feats'][start_id:start_id + len(batch_img_ids)] = pt.to_numpy(feats)



















