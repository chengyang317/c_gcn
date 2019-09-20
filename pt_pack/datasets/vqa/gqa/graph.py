# coding=utf-8
from pt_pack.datasets.base import Dataset
import json
import tqdm
import logging
import collections
from typing import List
from pt_pack.utils import to_path, to_tensor, get_idx
from pt_pack.datasets.field import ProxyField
from pt_pack.datasets.reader import H5Reader, DictReader, SwitchField
from pt_pack.datasets.vocab import Vocab
import torch
import dill
import numpy as np
import torch.nn.functional as F


logger = logging.getLogger(__name__)


__all__ = ['GraphGqaDataset']


def tokenize_gqa(sentence,
                 ignoredPunct=["?", "!", "\\", "/", ")", "("],
                 keptPunct=[".", ",", ";", ":"]):
    sentence = sentence.lower()
    for punct in keptPunct:
        sentence = sentence.replace(punct, " " + punct + " ")
    for punct in ignoredPunct:
        sentence = sentence.replace(punct, "")
    tokens = sentence.split()
    return tokens


class GraphGqaDataset(Dataset):

    name = 'graph'
    valid_field_names = ('q_ids', 'img_ids', 'q_tokens', 'img_names', 'img_shapes', 'a_tokens', 'a_counts', 'a_scores',
                         'q_lens', 'img_boxes', 'img_obj_feats', 'q_labels', 'a_labels')

    def __init__(self,
                 data_dir: str = 'work_dir/data/gqa',
                 split: str = 'train',
                 req_field_names: List[str] = ('img_obj_feats', 'q_labels', 'q_lens', 'a_labels', 'q_ids',
                                               'img_obj_nums', 'img_obj_boxes'),
                 is_lazy: bool = True
                 ):
        self._question_vocab = None
        self._answer_vocab = None
        if split == 'test':
            req_field_names = [item for item in req_field_names if item not in ('a_label_scores', 'a_label_counts')]
        super().__init__(data_dir, split, req_field_names, is_lazy=is_lazy)

    def build_fields(self):
        ann_reader = DictReader(self.load_combined_anns(self.data_dir, self.split))
        img_id_field = ann_reader['img_ids']
        fields = list()
        fields.extend([ann_reader[name] for name in ('q_ids', 'q_tokens', 'img_ids', 'q_labels', 'q_lens', 'a_labels')])

        if self.req_field_names is None or 'img_obj_feats' in self.req_field_names:
            obj_infos = json.load(self.data_dir.joinpath('objects/gqa_objects_info.json').open())
            h5_files = self.data_dir.joinpath('objects').glob('gqa_objects_*.h5')
            h5_readers = {h5_file.name: H5Reader(h5_file) for h5_file in h5_files}

            def idx_map_fn(dset_id):
                img_id = img_id_field[dset_id]
                obj_info = obj_infos[img_id]
                return obj_info['idx']
            for reader in h5_readers.values():
                for field in reader.fields:
                    field.idx_map_fn = idx_map_fn

            def switch_fn(dset_id, fields):
                img_id = img_id_field[dset_id]
                obj_info = obj_infos[img_id]
                return fields[f'gqa_objects_{obj_info["file"]}.h5']

            def depend_fn(x):
                return torch.from_numpy(x)

            feat_fields = {key: ProxyField('features', reader['features'], depend_fn) for key, reader in h5_readers.items()}

            def depend_fn(box, img_shape):
                box = torch.from_numpy(box)
                return box / torch.cat((img_shape, img_shape))

            box_fields = {key: ProxyField('boxes', (reader['bboxes'], ann_reader['img_shapes']), depend_fn) for key, reader in h5_readers.items()}

            fields.append(SwitchField('img_obj_feats', feat_fields, switch_fn))
            fields.append(SwitchField('img_obj_boxes', box_fields, switch_fn))
            fields.append(ann_reader['img_obj_nums'])

        return fields

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
    def q_tokenize(cls, text):
        return tokenize_gqa(text)

    @classmethod
    def load_combined_vocab(cls, data_dir='work_dir/data/gqa'):
        data_dir = to_path(data_dir).joinpath(cls.name)
        q_vocab_file = data_dir.joinpath('q_vocab.json')
        a_vocab_file = data_dir.joinpath('a_vocab.json')
        if not q_vocab_file.exists():
            logger.info(f'Creating question vocab {q_vocab_file.name}')
            q_vocab = Vocab.build_from_txt('q_vocab', data_dir.joinpath('vocabulary_gqa.txt'))
            q_vocab.save(q_vocab_file)
        else:
            q_vocab = Vocab.from_file('q_vocab', q_vocab_file)
        if not a_vocab_file.exists():
            logger.info(f'Creating answer vocab {a_vocab_file.name}')
            a_vocab = Vocab.build_from_txt('a_vocab', data_dir.joinpath('answers_gqa.txt'))
            a_vocab.save(a_vocab_file)
        else:
            a_vocab = Vocab.from_file('a_vocab', a_vocab_file)
        q_vocab.embed_init = np.load(data_dir.joinpath('gloves_gqa_no_pad.npy'))
        return q_vocab, a_vocab

    @classmethod
    def load_combined_anns(cls, data_dir: str = 'work_dir/data/gqa', data_split: str = 'train', q_max=30):
        data_dir = to_path(data_dir).joinpath(cls.name)
        ann_file = data_dir.joinpath(f'{data_split}_ann.pt')
        if ann_file.exists():
            return torch.load(ann_file)

        logger.info(f'Creating combined annotation files {ann_file.name}')
        q_vocab, a_vocab = cls.load_combined_vocab(data_dir.parent)
        if data_split in ('train', 'val'):
            q_file = data_dir.parent.joinpath(f'questions/{data_split}_balanced_questions.json')
            obj_infos = json.load(data_dir.parent.joinpath('objects/gqa_objects_info.json').open())
            # scene_file = data_dir.joinpath(f'sceneGraphs/{data_split}_sceneGraphs.json')
            q_anns = json.load(q_file.open())
        else:
            raise NotImplementedError()

        anns = collections.defaultdict(list)
        pbar = tqdm.tqdm(range(len(q_anns)), desc=f'Creating combined annotation files {ann_file.name}')
        for q_id, q_ann in q_anns.items():
            anns['splits'].append(data_split)
            anns['q_ids'].append(q_id)
            anns['img_ids'].append(q_ann['imageId'])
            anns['q_tokens'].append(cls.q_tokenize(q_ann['question']))
            q_label = [q_vocab[token] for token in anns['q_tokens'][-1][:q_max]]
            anns['q_lens'].append(len(q_label))
            q_label_t = torch.empty(q_max).fill_(q_vocab.padding_idx).long()
            q_label_t[:len(q_label)] = torch.tensor(q_label).long()
            anns['q_labels'].append(q_label_t)
            obj_info = obj_infos[q_ann['imageId']]
            anns['img_shapes'].append(torch.tensor((obj_info['width'], obj_info['height'])).float())
            anns['img_obj_nums'].append(torch.tensor(obj_info['objectsNum']))
            if 'answer' in q_ann:
                anns['a_tokens'].append(q_ann['answer'])
                anns['a_labels'].append(torch.tensor(a_vocab[q_ann['answer']]))
            pbar.update(1)
        anns_t = {key: to_tensor(value) for key, value in anns.items()}
        torch.save(anns_t, ann_file, pickle_module=dill)
        return anns_t

    def __len__(self):
        if self._length is None:
            ann_reader = DictReader(self.load_combined_anns(self.data_dir, self.split))
            self._length = len(ann_reader['q_ids'])
        return self._length

    def collate_fn(self, batch):
        if 'q_lens' in self.req_field_names:
            q_len_id = get_idx(self.req_field_names, 'q_lens')
            batch = sorted(batch, key=lambda x: x[q_len_id], reverse=True)
        samples = super().collate_fn(batch)
        return samples