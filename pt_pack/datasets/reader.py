# coding=utf-8
import h5py
from typing import List, Dict
from pathlib import Path
import logging
import json
from pt_pack.utils import to_seq, to_path
from pt_pack.datasets.field import Field, FieldGroup, ProxyField, SwitchField
from collections import defaultdict
from PIL import Image
import re
import torch

logger = logging.getLogger(__name__)


__all__ = ['DictReader', 'JsonReader', 'H5Reader', 'ImgDirReader', 'SwitchReader']


class Reader(object):
    def __init__(self, req_field_names=None, is_lazy=False):
        self.req_field_names = to_seq(req_field_names)
        self.is_lazy = is_lazy
        self.fields = None
        self.field_group = None
        self.has_init = False
        if not is_lazy:
            self.init_fields()
            self.has_init = True

    def init_fields(self):
        self.fields = self.build_fields()
        self.req_field_names = self.req_field_names or self.field_names
        self.fields = [self.name2fields[name] for name in self.req_field_names]
        self.field_group = FieldGroup(self.fields)
        self.has_init = True

    @property
    def is_dirty(self):
        return any(field.is_dirty for field in self.fields)

    @property
    def field_names(self):
        return [field.name for field in self.fields]

    @property
    def name2fields(self):
        return {field.name: field for field in self.fields}

    def build_fields(self) -> List[Field]:
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.field_group[item]
        # assert isinstance(item, str) and item in self.field_names
        return self.name2fields[item]

    def __iter__(self):
        return iter(self.field_group)

    def __len__(self):
        return len(self.field_group)

    def sync(self):
        if self.is_dirty:
            for field in self.fields:
                field.sync()
            assert not self.is_dirty

    def dict_read(self, datas):
        fields = []
        for key, value in datas.items():
            if self.req_field_names is None or key in self.req_field_names:
                fields.append(Field(key, value))
        return fields

    def list_read(self, datas):
        dict_datas = defaultdict(list)
        for row_data in datas:
            for key, value in row_data.items():
                dict_datas[key].append(value)
        return self.dict_read(dict_datas)

    def change_field_names(self, field_name_maps):
        for field in self.fields:
            if field.name in field_name_maps:
                field.change_name(field_name_maps[field.name])
        return self


class SwitchReader(Reader):
    def __init__(self, readers: Dict[str, Reader], switch_fn, req_field_names):
        self.readers = readers
        self.switch_fn = switch_fn
        super().__init__(req_field_names)

    def build_fields(self) -> List[Field]:
        assert len(self.readers) > 1
        field_names = self.readers.values()[0].field_names
        for reader in self.readers.values():
            assert set(field_names) == set(reader.field_names)
        fields = list()
        for field_name in field_names:
            option_fields = {key: reader[field_name] for key, reader in self.readers.items()}
            fields.append(SwitchField(field_name, option_fields, self.switch_fn))
        return fields


class DictReader(Reader):
    def __init__(self, data, req_field_names=None):
        self.dict_data = data
        super().__init__(req_field_names)

    def build_fields(self):
        return self.dict_read(self.dict_data)


class FileReader(Reader):
    def __init__(self, file_path, req_field_names=None):
        self.file_path = Path(file_path)
        super().__init__(req_field_names)

    def build_fields(self):
        logger.info(f'Building fields from file {self.file_path}')
        datas = self.load_file()
        assert isinstance(datas, (dict, list, h5py.File)), \
            f"Now do not support data types other than dict and list h5py.File type"
        if isinstance(datas, (dict, h5py.File)):
            return self.dict_read(datas)
        else:
            return self.list_read(datas)

    def load_file(self):
        raise NotImplementedError()


class H5Reader(FileReader):
    def __init__(self, h5_file, req_field_names=None, is_load=False):
        self.is_fast = is_load
        super().__init__(h5_file, req_field_names)

    def load_file(self):
        datas = h5py.File(self.file_path, 'r')
        if self.is_fast:
            datas = {key: torch.from_numpy(datas[key].value) for key in datas.keys()}
        return datas


class JsonReader(FileReader):
    def __init__(self, json_file: Path, req_field_names=None):
        super().__init__(json_file, req_field_names)

    def load_file(self):
        return json.load(self.file_path.open())


# class PklReader(Reader):
#     def __init__(self, pkl_file: Path, req_field_names=None):
#         super().__init__(req_field_names, data_file=pkl_file)
#
#     def build_fields(self, req_field_names, init_fields, data_file):
#         fields = self.build_fields_from_file(data_file, req_field_names)
#         return fields
#
#     def build_fields_from_file(self, data_file, req_field_names):
#         logger.info(f'Loading pkl file {data_file}')
#         pkl_data = pickle.load(data_file.open('rb'))
#         assert isinstance(pkl_data, dict), f"Now can not support other data type as {type(pkl_data)} but dict type"
#         if req_field_names is not None:
#             assert all([name in pkl_data.keys() for name in req_field_names])
#         fields = []
#         for key, value in pkl_data.items():
#             if req_field_names is None or key in req_field_names:
#                 if not isinstance(value, (list, tuple)):
#                     fields.append(Field(key, datas=value))
#                 else:
#                     fields.append(SeqField(key, datas=value))
#         return {field.name: field for field in fields}

#


class ImgDirReader(Reader):
    valid_field_names = ('img_ids', 'img_files', 'images')

    def __init__(self,
                 img_dir,
                 req_field_names=None,
                 file2idx_fn=None,
                 img_size=None,
                 transforms=None
                 ):
        self.img_dir = to_path(img_dir)
        assert self.img_dir.exists()
        self.file2idx_fn = file2idx_fn or self._file2idx_fn
        self.transforms = transforms or self.load_transforms(img_size)
        super().__init__(req_field_names)

    @staticmethod
    def load_transforms(img_size):
        from torchvision import transforms
        tf_components = [transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])]
        if img_size is not None:
            tf_components.insert(0, transforms.Resize(img_size))
        return transforms.Compose(tf_components)

    @staticmethod
    def _file2idx_fn(file):
        return int(re.findall('\d+', file.stem)[0])

    def build_fields(self):
        if self.req_field_names is not None:
            assert all([name in self.valid_field_names for name in self.req_field_names])
        fields = []
        img_files = self._get_image_files()
        img_file_field = Field('img_files', datas=img_files)
        fields.append(img_file_field)
        if self.req_field_names is None or 'img_ids' in self.req_field_names:
            img_idxs = [self.file2idx_fn(img_file) for img_file in img_files]
            fields.append(Field('img_ids', datas=img_idxs))
        if self.req_field_names is None or 'images' in self.req_field_names:
            fields.append(ProxyField('images', img_file_field, self._load_image))
        return fields

    def _get_image_files(self, order=True):
        img_files = list(self.img_dir.glob('*.jpg'))
        if len(img_files) == 0:
            img_files = list(self.img_dir.glob('*.png'))
        assert len(img_files), f'Can not find image in dir {self.img_dir}'
        if order:
            img_files.sort(key=lambda img_file: self.file2idx_fn(img_file))
        return img_files

    def _load_image(self, img_file):
        # img = imread(img_file, mode='RGB')
        img = Image.open(img_file).convert('RGB')
        img = self.transforms(img)
        return img[:3, :, :]



#
#
# class FieldReader(Reader):
#     def __init__(self, init_fields):
#         super().__init__(init_fields=init_fields)
#
#     def build_fields(self, req_field_names, init_fields, data_file):
#         return init_fields
#
#     @classmethod
#     def build_reader(cls, dict_data):
#         init_fields = [SeqField(name=key, datas=value) for key, value in dict_data.items()]
#         return cls(init_fields=init_fields)
#
#

#
#
# class ImgH5Reader(H5Reader):
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  req_field_names=('indexes_for_image', 'images_h5'),
#                  img_size=None,
#                  is_load=False
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         self.img_size = img_size
#         self.h5_file = ImgH5Writer.get_h5_file(self.data_dir, data_split, img_size)
#         if not self.h5_file.exists():
#             image_h5_writer = ImgH5Writer(self.data_dir, data_split, img_size)
#             image_h5_writer()
#         super().__init__(self.h5_file, req_field_names, is_load)
#
#
# class ImgFeatH5Reader(H5Reader):
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  req_field_names=('indexes_for_image', 'image_features_h5'),
#                  net_cfg=None,
#                  img_size=None,
#                  is_load=False
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         assert net_cfg is not None
#         self.net_cfg = net_cfg
#         self.img_size = img_size
#         self.h5_file = ImgFeatH5Writer.get_h5_file(self.data_dir, data_split, net_cfg, img_size)
#         init_fields = None
#         if not self.h5_file.exists():
#             writer = ImgFeatH5Writer(self.data_dir, data_split, net_cfg, img_size)
#             init_fields = writer()
#         super().__init__(self.h5_file, req_field_names, is_load, init_fields)
#

#
# class QestH5Reader(H5Reader):
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  req_field_names=('questions_h5', 'question_lengths'),
#                  special_tokens=None,
#                  add_start_token=True,
#                  add_end_token=False,
#                  is_load=True,
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         self.special_tokens = special_tokens
#         self.h5_file = QestH5Writer.get_h5_file(self.data_dir, data_split, add_start_token, add_end_token)
#         if not self.h5_file.exists():
#             writer = QestH5Writer(self.data_dir, data_split, special_tokens, add_start_token, add_end_token)
#             writer()
#         super().__init__(self.h5_file, is_load=is_load, req_field_names=req_field_names)
#
#     @property
#     def vocab(self):
#         return QestReader.load_vocab(self.data_dir, self.special_tokens)
#
#     def build_fields(self, req_field_names, init_fields, data_file):
#         fields = super().build_fields(req_field_names, init_fields, data_file)
#         for field in fields.values():
#             if self.is_fast:
#                 field.datas = field.datas.long()
#             else:
#                 field.add_elem_hook(lambda x: x.astype(np.long))
#         return fields
#
#
# class QestReader(Reader):
#     special_tokens = ('<NULL>', '<START>', '<END>', '<UNK>')
#
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  req_field_names=('questions', 'question_lengths'),
#                  special_tokens=None,
#                  add_start_token=True,
#                  add_end_token=False,
#                  init_fields=None,
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         # self.special_tokens = special_tokens or self.special_tokens
#         json_file = self.data_dir.joinpath(f'pt_pack/{data_split}_ann.json')
#         self.add_start_token = add_start_token
#         self.add_end_token = add_end_token
#         self.vocab = self.load_vocab(self.data_dir, special_tokens)
#         super().__init__(req_field_names, init_fields, json_file)
#
#     def build_fields(self, req_field_names, init_fields: Dict[str, SeqField], data_file):
#         f_names = ('question_tokens', 'question_token_lengths')
#         if init_fields is not None:
#             assert all(f_name in init_fields for f_name in f_names)
#             q_field, q_l_field = [init_fields[f_name] for f_name in f_names]
#         else:
#             reader = JsonReader(data_file, req_field_names=f_names)
#             q_field, q_l_field = reader.fields
#         max_len = max(q_l_field.data) + self.add_start_token + self.add_end_token
#
#         def _encode(q_tokens: list):
#             q_idxs = []
#             if self.add_start_token:
#                 q_idxs.append(self.vocab.stoi['<START>'])
#             for token in q_tokens:
#                 if token not in self.vocab.stoi:
#                     raise KeyError('Token "%s" not in vocab' % token)
#                 q_idxs.append(self.vocab.stoi[token])
#             if self.add_end_token:
#                 q_idxs.append(self.vocab.stoi['<END>'])
#             while len(q_idxs) < max_len:
#                 q_idxs.append(self.vocab.stoi['<NULL>'])
#             return torch.tensor(q_idxs)
#         q_field = q_field.clone('questions')
#         q_field.add_elem_hook(_encode)
#         q_l_field = q_l_field.clone('question_lengths')
#         q_l_field.add_elem_hook(lambda q_len: q_len + self.add_start_token + self.add_end_token)
#         return {q_field.name: q_field, q_l_field.name: q_l_field}
#
#     @classmethod
#     def load_vocab(cls, data_dir: Path, specials=None):
#         specials = specials or cls.special_tokens
#         counter_file = data_dir.joinpath('pt_pack/counter.json')
#         assert counter_file.exists(), f'Can not find counter file {counter_file}'
#         q_counter = json.load(counter_file.open())['question_counter']
#         for special_token in specials:
#             if special_token not in q_counter:
#                 q_counter[special_token] = 1
#         q_vocab = Vocab(q_counter, specials=specials)
#         # a_vocab = Vocab(counter_data['answer_counter'], specials=specials)
#         return q_vocab
#
#
# class QestH5Writer(H5Writer):
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  special_tokens=None,
#                  add_start_token=True,
#                  add_end_token=True,
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         self.special_tokens = special_tokens
#         self.add_start_token = add_start_token
#         self.add_end_token = add_end_token
#         h5_file = self.get_h5_file(self.data_dir, data_split, add_start_token, add_end_token)
#         super().__init__(h5_file)
#
#     @classmethod
#     def get_h5_file(cls, data_dir, data_split, add_start_token, add_end_token):
#         return data_dir.joinpath(f'pt_pack/{data_split}_questions_{add_start_token}_{add_end_token}.h5')
#
#     def build_reader(self):
#         reader = QestReader(self.data_dir, self.data_split, ['questions', 'question_lengths'], self.special_tokens,
#                             self.add_start_token, self.add_end_token)
#         q_field, q_l_field = reader.fields
#         q_field = q_field.clone('questions_h5')
#         return FieldReader([q_field, q_l_field])
#
#
# class QestFeatH5Reader(H5Reader):
#
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  req_field_names=('question_features_h5', 'question_lengths'),
#                  special_tokens=None,
#                  add_start_token=True,
#                  add_end_token=True,
#                  vector_name='glove.6B.200d'
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         h5_file = QestFeatH5Writer.get_h5_file(self.data_dir, data_split, add_start_token, add_end_token, vector_name)
#         self.special_tokens = special_tokens
#         if not h5_file.exists():
#             writer = QestFeatH5Writer(self.data_dir, data_split, special_tokens, add_start_token, add_end_token, vector_name)
#             writer()
#         super().__init__(h5_file, req_field_names, True)
#
#     @property
#     def vocab(self):
#         return QestReader.load_vocab(self.data_dir, self.special_tokens)
#
#     def build_fields(self, req_field_names, init_fields, data_file):
#         fields = super().build_fields(req_field_names, init_fields, data_file)
#         for field in fields.values():
#             if field.name == 'question_lengths':
#                 if self.is_fast:
#                     field.datas = field.datas.long()
#                 else:
#                     field.add_hook(lambda x: x.astype(np.long))
#         return fields
#
#
# class QestFeatH5Writer(H5Writer):
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  special_tokens,
#                  add_start_token=True,
#                  add_end_token=True,
#                  vector_name='glove.6B.300d'
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         self.special_tokens = special_tokens
#         self.add_start_token = add_start_token
#         self.add_end_token = add_end_token
#         self.vector_name = vector_name
#         h5_file = self.get_h5_file(self.data_dir, data_split, add_start_token, add_end_token, vector_name)
#         super().__init__(h5_file)
#
#     @classmethod
#     def get_h5_file(cls, data_dir, data_split, add_start_token, add_end_token, vector_name):
#         file_name = f'{data_split}_question_features_{vector_name}_{add_start_token}_{add_end_token}.h5'
#         return data_dir.joinpath(f'pt_pack/{file_name}')
#
#     def build_reader(self):
#         reader = QestReader(self.data_dir, self.data_split, ('questions', 'question_lengths'), self.special_tokens,
#                             self.add_start_token, self.add_end_token)
#         q_field, q_l_field = reader.fields
#         vocab = reader.vocab
#         vocab.load_vectors(self.vector_name, unk_init=torch.nn.init.normal_, cache=os.path.expanduser('~/.vector_cache'))
#
#         def _feat_func(q_idxs):
#             q_vectors = [vocab.vectors[q_idx] for q_idx in q_idxs]
#             return torch.stack(q_vectors)
#         q_field = q_field.clone('question_features_h5')
#         q_field.add_elem_hook(_feat_func)
#         return FieldReader([q_field, q_l_field])
#
#
# class AnswerReader(Reader):
#     special_tokens = ('<UNK>',)
#
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  req_field_names=('answers',),
#                  special_tokens=None,
#                  init_fields=None
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         # self.special_tokens = special_tokens or self.special_tokens
#         json_file = self.data_dir.joinpath(f'pt_pack/{data_split}_ann.json')
#         self.vocab = self.load_vocab(self.data_dir, special_tokens)
#         super().__init__(req_field_names, init_fields=init_fields, data_file=json_file)
#
#     def build_fields(self, req_field_names, init_fields: Dict[str, SeqField], data_file):
#         if init_fields is not None:
#             a_field = init_fields['answer_tokens']
#         else:
#             reader = JsonReader(data_file, req_field_names=('answer_tokens'))
#             a_field = reader.fields[0]
#
#         def _encode(tokens: list):
#             idxs = []
#             for token in tokens:
#                 if token not in self.vocab.stoi:
#                     raise KeyError('Token "%s" not in vocab' % token)
#                 idxs.append(self.vocab.stoi[token])
#             return idxs[0]
#         a_field = a_field.clone('answers')
#         a_field.elem_map_funcs.append(_encode)
#         return {a_field.name: a_field}
#
#     @classmethod
#     def load_vocab(cls, data_dir: Path, specials=None):
#         specials = specials or cls.special_tokens
#         counter_file = data_dir.joinpath('pt_pack/counter.json')
#         assert counter_file.exists(), f'Can not find counter file {counter_file}'
#         counter = json.load(counter_file.open())['answer_counter']
#         for special_token in specials:
#             if special_token not in counter:
#                 counter[special_token] = 1
#         vocab = Vocab(counter, specials=specials)
#         # a_vocab = Vocab(counter_data['answer_counter'], specials=specials)
#         return vocab
#
#
# class AnswerFeatH5Reader(H5Reader):
#
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  req_field_names=('answer_features_h5',),
#                  special_tokens=None,
#                  vector_name='glove.6B.300d'
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         h5_file = AnswerFeatH5Writer.get_h5_file(self.data_dir, data_split, vector_name)
#         self.special_tokens = special_tokens
#         if not h5_file.exists():
#             writer = AnswerFeatH5Writer(self.data_dir, data_split, special_tokens, vector_name)
#             writer()
#         super().__init__(h5_file, req_field_names, True)
#
#     @property
#     def vocab(self):
#         return AnswerReader.load_vocab(self.data_dir, self.special_tokens)
#
#
# class AnswerFeatH5Writer(H5Writer):
#     def __init__(self,
#                  data_dir,
#                  data_split,
#                  special_tokens,
#                  vector_name='glove.6B.300d'
#                  ):
#         self.data_dir = Path(data_dir)
#         self.data_split = data_split
#         self.special_tokens = special_tokens
#         self.vector_name = vector_name
#         h5_file = self.get_h5_file(self.data_dir, data_split, vector_name)
#         super().__init__(h5_file)
#
#     @classmethod
#     def get_h5_file(cls, data_dir, data_split, vector_name):
#         return data_dir.joinpath(f'pt_pack/{data_split}_answer_features_{vector_name}.h5')
#
#     def build_reader(self):
#         reader = AnswerReader(self.data_dir, self.data_split, ('answers',), self.special_tokens)
#         field: Field = reader['answers']
#         vocab = reader.vocab
#         vocab.load_vectors(self.vector_name, unk_init=torch.nn.init.normal_)
#
#         def _feat_func(idxs):
#             vectors = [vocab.vectors[idx] for idx in idxs]
#             return torch.stack(vectors)
#         field = field.clone('answer_features_h5')
#         field.elem_map_funcs.append(_feat_func)
#         return FieldReader([field])















