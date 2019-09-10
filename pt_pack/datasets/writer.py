# coding=utf-8
from typing import List
from .field import Field
from ..utils import to_path
import json
import logging
import h5py
from tqdm import tqdm
import torch
import numpy as np

logger = logging.getLogger(__name__)


__all__ = ['H5Writer', 'JsonWriter']


class Writer(object):
    def __init__(self, ):
        self.fields = self.build_fields()

    def build_fields(self) -> List[Field]:
        raise NotImplementedError()

    def write(self):
        raise NotImplementedError()


class FileWriter(Writer):
    def __init__(self, file_path):
        self.file_path = to_path(file_path)
        super().__init__()

    def write(self):
        for field in self.fields:
            field.sync()
        self.write_file({field.name: field.datas for field in self.fields})
        return self.fields

    def write_file(self, datas):
        raise NotImplementedError()

    def __call__(self):
        if self.file_path.exists():
            logger.info(f'{self.file_path.name} exists.')
        else:
            return self.write()


class JsonWriter(FileWriter):
    def __init__(self, json_file):
        super().__init__(json_file)

    def write_file(self, datas):
        json.dump(datas, self.file_path.open('w'))


class H5Writer(FileWriter):
    def __init__(self, h5_file, batch_size=64):
        self.batch_size = batch_size
        super().__init__(h5_file)

    def group_data(self, batch_data):
        if isinstance(batch_data, (list, tuple)):
            if isinstance(batch_data[0], torch.Tensor):
                return torch.stack(batch_data).cpu().numpy()
            elif isinstance(batch_data[0], np.ndarray):
                return np.stack(batch_data)
            else:
                batch_data = [np.asarray(dset_data) for dset_data in batch_data]
                return np.stack(batch_data)
        elif isinstance(batch_data, torch.Tensor):
            return batch_data.cpu().numpy()
        else:
            assert isinstance(batch_data, np.ndarray), f'dset_datas type is {type(batch_data)}'
            return batch_data

    def write(self):
        with h5py.File(self.file_path, 'w') as h5_fd:
            dset_len = len(self.fields[0])
            dsets = [None] * len(self.fields)
            batch_ids = []
            for dset_id in tqdm(range(dset_len)):
                batch_ids.append(dset_id)
                if len(batch_ids) == self.batch_size:
                    for idx, field in enumerate(self.fields):
                        batch_data = field.load_batch_data(batch_ids)
                        batch_data = self.group_data(batch_data)
                        if dsets[idx] is None:
                            dsets[idx] = h5_fd.create_dataset(field.name, (dset_len, *batch_data[0].shape))
                        dsets[idx][batch_ids[0]:batch_ids[-1]+1] = batch_data
                    batch_ids = []
            if len(batch_ids) > 0:
                for idx, field in enumerate(self.fields):
                    batch_data = field.load_batch_data(batch_ids)
                    batch_data = self.group_data(batch_data)
                    dsets[idx][batch_ids[0]:batch_ids[-1] + 1] = batch_data
        return self.fields



# class PklWriter(Writer):
#     def __init__(self, pkl_file):
#         super().__init__(pkl_file)
#
#     def write(self):
#         self.fields.sync()
#         pkl_data = {field.name: field.datas for field in self.fields.fields}
#         pickle.dump(pkl_data, self.data_file.open('wb'))
#         return self.fields.fields


# class ImgH5Writer(H5Writer):
#     _supported_names = ('indexes_for_image', 'images_h5')
#
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  img_size=None,
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         self.img_size = img_size
#         h5_file = self.get_h5_file(self.data_dir, data_split, img_size)
#         super().__init__(h5_file)
#
#     @classmethod
#     def get_h5_file(cls, data_dir, data_split, img_size):
#         size_attr = '_'.join([str(item) for item in img_size]) if img_size is not None else 'origin'
#         return data_dir.joinpath('pt_pack').joinpath(f'{data_split}_images_{size_attr}.h5')
#
#     def build_fields(self):
#         image_reader = ImgReader(self.data_dir, self.data_split, img_size=self.img_size)
#         img_idx_field = image_reader['indexes_for_image']
#         img_field = image_reader['images']
#         img_field = img_field.clone('images_h5')
#         return FieldReader([img_idx_field, img_field])


# class ImgFeatH5Writer(H5Writer):
#     _supported_names = ('indexes_for_image', 'image_features_h5')
#
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  net_cfg,
#                  img_size=None,
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         self.img_size = img_size
#         self.net_cfg = net_cfg
#         self.feat_net = get_feat_net(net_cfg)
#         self.feat_net.eval()
#         self.feat_net.cuda()
#         h5_file = self.get_h5_file(self.data_dir, data_split, net_cfg, img_size)
#         super().__init__(h5_file)
#
#     @classmethod
#     def get_h5_file(cls, data_dir, data_split, net_cfg, img_size):
#         """
#
#         :param data_dir:
#         :param data_split:
#         :param net_cfg: {net_cls, net_opt: {pre_trained, requeired_layer_idxs}}
#         :param img_size:
#         :return:
#         """
#         def var2str(var):
#             var_attr = []
#             if isinstance(var, (tuple, list)):
#                 for item in var:
#                     var_attr.extend(var2str(item))
#             elif isinstance(var, dict):
#                 for item in var.values():
#                     var_attr.extend(var2str(item))
#             else:
#                 var_attr.append(str(var))
#             return var_attr
#         net_attr = '_'.join(var2str(net_cfg))
#         size_attr = '_'.join([str(item) for item in img_size]) if img_size is not None else 'origin'
#         return data_dir.joinpath('pt_pack').joinpath(f'{data_split}_features_{net_attr}_{size_attr}.h5')
#
#     def build_fields(self):
#
#         def _map_func(batch_imgs):
#             img = torch.stack(batch_imgs)
#             img = img.cuda()
#             with torch.no_grad():
#                 return self.feat_net(img)
#
#         image_reader = ImgReader(self.data_dir, self.data_split, img_size=self.img_size)
#         img_idxs_field = image_reader['indexes_for_image']
#         feats_field = image_reader['images'].clone('image_features_h5')
#         feats_field.map_funcs.append(_map_func)
#         return FieldReader([img_idxs_field, feats_field])
