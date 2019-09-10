# coding=utf-8
import json
from pt_pack.datasets.field import ProxyField, PseudoField
from pt_pack.datasets.reader import DictReader
from pt_pack.datasets.base import Dataset
import pt_pack.utils as utils
from pt_pack.datasets.vocab import Vocab
from typing import List
import logging
import torch
import collections
from tqdm import tqdm
import zarr
import numpy as np


logger = logging.getLogger(__name__)


__all__ = ['CgsVqa2Dataset']


class CgsVqa2Dataset(Dataset):
    """
    Reimplement the dataset for paper: Learning Conditioned Graph Structures for Interpretable Visual Question Answering
    """

    name = 'cgs_vqa2'
    valid_field_names = ('q_ids', 'img_ids', 'q_tokens', 'img_names', 'img_shapes', 'a_tokens', 'a_counts', 'a_scores',
                         'q_lens', 'img_obj_feats', 'img_obj_feats', 'q_labels', 'a_label_scores', 'a_label_counts')

    def __init__(self,
                 data_dir: str = 'work_dir/data/vqa2',
                 split: str = 'train',
                 req_field_names: List[str] = ('img_obj_feats', 'q_labels', 'q_lens', 'a_label_counts',
                                               'a_label_scores', 'q_ids'),
                 batch_size: int = 64,
                 workers_num: int = 4,
                 shuffle: bool = None,
                 ):
        self._question_vocab = None
        self._answer_vocab = None
        if split == 'test':
            req_field_names = [item for item in req_field_names if item.split('_')[0] != 'a']
        super().__init__(data_dir, split, req_field_names, batch_size, workers_num, shuffle)

    @property
    def answer_vocab(self):
        if getattr(self, '_answer_vocab', None) is None:
            self._question_vocab, self._answer_vocab = self.load_combined_vocab(self.data_dir)
        return self._answer_vocab

    @property
    def question_vocab(self):
        if getattr(self, '_question_vocab', None) is None:
            self._question_vocab, self._answer_vocab = self.load_combined_vocab(self.data_dir)
        return self._question_vocab

    @classmethod
    def load_combined_vocab(cls, data_dir='work_dir/data/vqa2'):
        data_dir = utils.to_path(data_dir)
        q_vocab_file = data_dir.joinpath(f'{cls.name}/q_vocab.json')
        a_vocab_file = data_dir.joinpath(f'{cls.name}/a_vocab.json')
        if not q_vocab_file.exists():
            logger.info(f'Creating question vocab {q_vocab_file.name}')
            q_dict = json.load(data_dir.joinpath(cls.name, 'train_q_dict.json').open())
            id2words = ['<padding>'] + list(q_dict['itow'].values())
            word2ids = {word: idx for idx, word in enumerate(id2words)}
            q_vocab = Vocab.build_from_statistic(id2words, word2ids, 'q_vocab', data_dir.joinpath(f'{cls.name}'))
            q_vocab.save(q_vocab_file)
        if not a_vocab_file.exists():
            logger.info(f'Creating answer vocab {a_vocab_file.name}')
            a_dict = json.load(data_dir.joinpath(cls.name, 'train_a_dict.json').open())
            id2words = list(a_dict['itow'].values()) + ['<padding>']
            word2ids = {word: idx for idx, word in enumerate(id2words)}
            a_vocab = Vocab.build_from_statistic(id2words, word2ids, 'a_vocab', data_dir.joinpath(f'{cls.name}'))
            a_vocab.save(a_vocab_file)
        q_vocab = Vocab.from_file('q_vocab', data_dir.joinpath(cls.name), q_vocab_file)
        a_vocab = Vocab.from_file('a_vocab', data_dir.joinpath(cls.name), a_vocab_file)
        return q_vocab, a_vocab


    @classmethod
    def load_combined_anns(cls, data_dir: str = 'data/vqa2', split: str = 'train', ):
        data_dir = utils.to_path(data_dir)
        ann_file = data_dir.joinpath(f'{cls.name}/{split}_ann.pt')
        if ann_file.exists():
            anns_t = torch.load(ann_file)
        else:
            logger.info(f'Creating combined annotation files {ann_file.name}')
            q_vocab, a_vocab = cls.load_combined_vocab(data_dir)
            if split in ('train', 'val'):
                json_anns = json.load(data_dir.joinpath(cls.name, f'vqa_{split}_final_3000.json').open())
                ins_anns = json.load(data_dir.joinpath(f'annotations/instances_{split}2014.json').open())
                img_anns = {ann['id']: ann for ann in ins_anns['images']}
            elif split == 'test':
                json_anns = json.load(data_dir.joinpath(cls.name, f'vqa_test_toked.json').open())
                q_anns = json.load(data_dir.joinpath(f'annotations/v2_OpenEnded_mscoco_{split}2015_questions.json').open())['questions']
                ins_anns = json.load(data_dir.joinpath(f'annotations/image_info_{split}2015.json').open())
                img_anns = {ann['id']: ann for ann in ins_anns['images']}
            elif split == 'train_val':
                train_anns, val_anns = [cls.load_combined_anns(data_dir, split) for split in ('train', 'val')]
                anns_t = {}
                for key in train_anns.keys():
                    if torch.is_tensor(train_anns[key]):
                        anns_t[key] = torch.cat((train_anns[key], val_anns[key]), dim=0)
                    elif isinstance(train_anns[key], (list, tuple)):
                        anns_t[key] = list(train_anns[key]) + list(val_anns[key])
                    else:
                        raise NotImplementedError()
                return anns_t
            else:
                raise NotImplementedError()

            anns = collections.defaultdict(list)
            for idx, json_ann in enumerate(tqdm(json_anns)):
                anns['splits'].append(split)
                anns['q_ids'].append(json_ann['question_id'])
                anns['img_ids'].append(int(json_ann['image_id']))
                anns['q_tokens'].append(json_ann['question_toked'])
                anns['q_lens'].append(len(anns['q_tokens'][-1]))

                q_label = [q_vocab[token] for token in anns['q_tokens'][-1]]
                q_label_t = torch.empty(30).fill_(q_vocab.padding_idx).long()
                q_label_t[:len(q_label)] = torch.tensor(q_label).long()
                anns['q_labels'].append(q_label_t)
                img_ann = img_anns[int(json_ann['image_id'])]
                anns['img_names'].append(img_ann['file_name'])
                anns['img_shapes'].append((img_ann['width'], img_ann['height']))
                if split == 'test':
                    continue

                anns['a_counts'].append([a_count for a_count in json_ann['answers'] if a_count[0] in a_vocab.word2idx])
                anns['a_tokens'].append(json_ann['answer'])
                anns['a_scores'].append([a_score for a_score in json_ann['answers_w_scores'] if a_score[0] in a_vocab.word2idx])
            anns_t = {key: utils.to_tensor(value) for key, value in anns.items()}
            torch.save(anns_t, ann_file)
        return anns_t

    def __len__(self):
        return len(self['q_labels'])

    def collate_fn(self, batch):
        if 'q_lens' in self.req_field_names:
            q_len_id = utils.get_idx(self.req_field_names, 'q_lens')
            batch = sorted(batch, key=lambda x: x[q_len_id], reverse=True)
        return super().collate_fn(batch)

    def build_fields(self):
        ann_reader = DictReader(self.load_combined_anns(self.data_dir, self.split))
        fields = list()

        obj_fields = self.load_obj_fields(ann_reader['img_ids'], ann_reader['img_shapes'])
        if obj_fields is not None:
            fields.extend(obj_fields)
        fields.extend(self.load_ann_fields(ann_reader))
        return fields

    def load_obj_fields(self, img_ids_field, img_shapes_field):
        data_dir = self.data_dir.joinpath(self.name)
        if self.req_field_names is not None and utils.not_in(('img_obj_feats', 'img_box_feats'), self.req_field_names):
            return None

        if self.split != 'test':
            obj_feats = zarr.open(data_dir.joinpath('trainval.zarr').as_posix(), mode='r')
            box_feats = zarr.open(data_dir.joinpath('trainval_boxes.zarr').as_posix(), mode='r')
        else:
            obj_feats = zarr.open(data_dir.joinpath('test.zarr').as_posix(), mode='r')
            box_feats = zarr.open(data_dir.joinpath('test_boxes.zarr').as_posix(), mode='r')

        obj_field = PseudoField('img_obj_feats', obj_feats, lambda dset_id: str(img_ids_field[dset_id].item()))
        obj_field = ProxyField('img_obj_feats', obj_field, lambda x: torch.from_numpy(np.asarray(x)))
        box_field = PseudoField('img_box_feats', box_feats, lambda dset_id: str(img_ids_field[dset_id].item()))

        def depend_fn(box, img_shape):
            box = torch.from_numpy(np.asarray(box))
            img_shape = img_shape.unsqueeze(0)
            img_shape = torch.cat((img_shape, img_shape), dim=-1).float()
            box = box / img_shape
            return box

        box_field = ProxyField('img_box_feats', (box_field, img_shapes_field), depend_fn)

        def depend_fn(obj, box):
            return torch.cat((obj, box), dim=-1)
        obj_field = ProxyField('img_obj_feats', (obj_field, box_field), depend_fn)
        return [obj_field, box_field]

    def load_ann_fields(self, ann_reader):
        fields = list()
        fields.extend(ann_reader.fields)
        if self.split == 'test':
            return fields

        def depend_fn(a_scores):
            a_label_score = torch.empty(len(self.answer_vocab)).fill_(0).float()
            for a_token, a_score in a_scores:
                a_label_score[self.answer_vocab[a_token]] = a_score
            return a_label_score
        fields.append(ProxyField('a_label_scores', ann_reader['a_scores'], depend_fn=depend_fn))

        def depend_fn(answers):
            a_label_count = torch.empty(len(self.answer_vocab)).fill_(0).float()
            for a_token, a_count in answers:
                a_label_count[self.answer_vocab[a_token]] = a_count
            return a_label_count
        fields.append(ProxyField('a_label_counts', ann_reader['a_counts'], depend_fn=depend_fn))

        return fields



