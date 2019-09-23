# coding=utf-8
from pt_pack.datasets.base import Dataset
from pt_pack.datasets.reader import DictReader, ProxyField, H5Reader
import logging
import json
from pt_pack.datasets.vocab import Vocab
from tqdm import tqdm
import nltk
import torch
from pt_pack.utils import to_path, get_idx, to_tensor
from collections import defaultdict
import numpy as np
from typing import List
from .images import ClevrImgH5Reader


sys_logger = logging.getLogger(__name__)


__all__ = ['ClevrDataset', 'GraphClevrDataset']



class ClevrDataset(Dataset):
    name = 'clevr'
    valid_field_names = ('q_ids', 'q_tokens', 'q_lens', 'q_labels', 'q_types', 'a_tokens', 'a_labels', 'img_ids',
                         'img_names', 'images', 'img_feats')

    def __init__(self,
                 data_dir: str='data/CLEVR_v1.0',
                 split: str= 'train',
                 req_field_names: List[str]=('images', 'q_labels', 'q_lens', 'a_labels', 'q_ids'),
                 is_load: bool=False,
                 ):
        self._question_vocab = None
        self._answer_vocab = None
        self.is_load = is_load
        super().__init__(data_dir, split, req_field_names)

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
        # if getattr(cls, 'q_tokenizer', None) is None:
        #     from spacy.tokenizer import Tokenizer
        #     import en_core_web_sm
        #     nlp = en_core_web_sm.load()
        #     setattr(cls, 'q_tokenizer', Tokenizer(nlp.vocab))
        # return [t.text if '?' not in t.text else t.text[:-1] for t in cls.q_tokenizer(text.lower())]
        tokens = nltk.word_tokenize(text.lower())
        return tokens[:-1]

    @classmethod
    def a_tokenize(cls, answer):
        return answer

    @classmethod
    def load_combined_vocab(cls, data_dir='data/CLEVR_v1.0'):
        data_dir = to_path(data_dir)
        q_vocab_file = data_dir.joinpath(f'{cls.name}/q_vocab.json')
        a_vocab_file = data_dir.joinpath(f'{cls.name}/a_vocab.json')
        if q_vocab_file.exists() and a_vocab_file.exists():
            q_vocab = Vocab.build_from_counter_file('q_vocab', data_dir.joinpath(cls.name), q_vocab_file,
                                                    special_symbols='<padding>')
            a_vocab = Vocab.build_from_counter_file('a_vocab', data_dir.joinpath(cls.name), a_vocab_file)
            return q_vocab, a_vocab

        ann_files = [data_dir.joinpath(f'questions/CLEVR_{split}_questions.json') for split in ('train', 'val')]
        q_anns = [json.load(a_file.open())['questions'] for a_file in ann_files]
        if not q_vocab_file.exists():
            questions = [ann_item['question'] for q_ann in q_anns for ann_item in q_ann]
            q_tokens = []
            for question in tqdm(questions, desc='Tokenizing questions'):
                q_tokens.append(cls.q_tokenize(question))
            q_vocab = Vocab.build_from_sequences('q_vocab', data_dir.joinpath(f'{cls.name}'), q_tokens,
                                                 special_symbols='<padding>')
            q_vocab.dump_to_file(q_vocab_file)
        else:
            q_vocab = Vocab.build_from_counter_file('q_vocab', data_dir.joinpath(cls.name), q_vocab_file,
                                                    special_symbols='<padding>')
        if not a_vocab_file.exists():
            answers = [ann_item['answer'] for ann in q_anns for ann_item in ann]
            a_tokens = []
            for answer in tqdm(answers, desc='Tokenizing answers'):
                a_tokens.append(cls.a_tokenize(answer))
            a_vocab = Vocab.build_from_sequences('a_vocab', data_dir.joinpath(f'{cls.name}'), a_tokens)
            a_vocab.dump_to_file(a_vocab_file)
        else:
            a_vocab = Vocab.build_from_counter_file('a_vocab', data_dir.joinpath(cls.name), a_vocab_file)
        q_vocab.strict = True
        a_vocab.strict = True
        return q_vocab, a_vocab

    @classmethod
    def load_combined_anns(cls, data_dir, data_split):
        data_dir = to_path(data_dir)
        ann_file = data_dir.joinpath(f'{cls.name}/{data_split}_ann.pt')
        if ann_file.exists():
            anns_t = torch.load(ann_file)
        else:
            sys_logger.info(f'Creating combined annotation files {ann_file.name}')
            q_vocab, a_vocab = cls.load_combined_vocab(data_dir)

            if data_split in ('valB_30k', 'valB_120k'):
                q_anns = json.load(data_dir.joinpath('questions', f'CLEVR_valB_questions.json').open())['questions']
                if data_split == 'valB_30k':
                    q_anns = q_anns[:30000]
                else:
                    q_anns = q_anns[30000:]
            else:
                q_anns = json.load(data_dir.joinpath('questions', f'CLEVR_{data_split}_questions.json').open())['questions']
            anns = defaultdict(list)
            for q_item in tqdm(q_anns, desc=f'Creating {ann_file.name}'):
                anns['q_ids'].append(q_item['question_index'])
                anns['img_ids'].append(q_item['image_index'])
                anns['img_names'].append(q_item['image_filename'])
                anns['q_tokens'].append(cls.q_tokenize(q_item['question']))
                anns['q_types'].append(q_item['question_family_index'])
                anns['a_tokens'].append(q_item['answer'])
                anns['q_lens'].append(len(anns['q_tokens'][-1]))
                q_label = [q_vocab[token] for token in anns['q_tokens'][-1]]
                q_label_t = torch.empty(47).fill_(q_vocab.padding_idx).long()
                q_label_t[:len(q_label)] = torch.tensor(q_label).long()
                anns['q_labels'].append(q_label_t)
                anns['a_labels'].append(a_vocab[q_item['answer']])
            anns_t = {name: to_tensor(value) for name, value in anns.items()}
            torch.save(anns_t, ann_file)
        return anns_t

    def build_fields(self):
        ann_reader = DictReader(self.load_combined_anns(self.data_dir, self.split))
        fields = list()
        fields.extend([ann_reader[f_name] for f_name in ('q_lens', 'a_labels', 'q_ids', 'q_labels')])
        img_idx_field = ann_reader['img_ids']

        if self.req_field_names is None or 'images' in self.req_field_names:
            img_reader = ClevrImgH5Reader(self.data_dir.joinpath('clevr'), self.split, is_load=self.is_load)
            img_field = ProxyField('images', img_reader['images'], depend_fn=lambda x: torch.from_numpy(x))
            # img_field = img_reader['images']
            img_id2field_id = {int(img_idx): field_idx for field_idx, img_idx in enumerate(img_reader['img_ids'])}

            def idx_map_fn(dset_idx):
                img_idx = img_idx_field[dset_idx].item()
                field_idx = img_id2field_id[int(img_idx)]
                return int(field_idx)
            for field in img_reader.fields:
                field.idx_map_fn = idx_map_fn
            fields.append(img_field)
        return fields

    def collate_fn(self, batch):
        if 'q_lens' in self.req_field_names:
            q_len_id = get_idx(self.req_field_names, 'q_lens')
            batch = sorted(batch, key=lambda x: x[q_len_id], reverse=True)
        batch_dict = super().collate_fn(batch)
        return batch_dict

    def __len__(self):
        return len(self['q_labels'])

    def evaluate(self, eval_log: dict):
        q_id2type_file = self.data_dir.joinpath(self.name, 'val_q_id2typ.json')
        if not q_id2type_file.exists():
            q_anns = json.load(self.data_dir.joinpath('questions/CLEVR_val_questions.json').open())['questions']
            q_id2type = {q_ann['question_index']: q_ann['program'][-1]['function'] for q_ann in q_anns}
            json.dump(q_id2type, q_id2type_file.open('w'))
        q_id2type = json.load(q_id2type_file.open())

        correct_by_q_type = defaultdict(list)
        for q_idx, is_true in eval_log.items():
            correct_by_q_type['Overall'].append(is_true)
            q_type = q_id2type[str(q_idx)]
            correct_by_q_type[q_type].append(is_true)
        acc_by_q_type = {}
        for q_type, vals in correct_by_q_type.items():
            vals = np.asarray(vals)
            # print(q_type, '%d / %d = %.2f' % (vals.sum(), vals.shape[0], 100.0 * vals.mean()))
            acc_by_q_type[q_type] = vals.mean()
        return acc_by_q_type


class GraphClevrDataset(Dataset):
    name = 'graph'
    valid_field_names = ('q_ids', 'q_tokens', 'q_lens', 'q_labels', 'q_types', 'a_tokens', 'a_labels', 'img_ids',
                         'img_names', 'images', 'img_obj_feats', 'img_obj_nums', 'img_obj_boxes')

    def __init__(self,
                 data_dir: str = 'work_dir/data/CLEVR_v1.0',
                 split: str = 'train',
                 req_field_names: List[str] = ('img_obj_feats', 'q_labels', 'q_lens', 'a_labels', 'q_ids',
                                               'img_obj_nums', 'img_obj_boxes'),
                 is_load: bool = False,
                 is_lazy: bool = False,
                 ):
        self._question_vocab = None
        self._answer_vocab = None
        self.is_load = is_load
        super().__init__(data_dir, split, req_field_names, is_lazy = is_lazy)

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
        # if getattr(cls, 'q_tokenizer', None) is None:
        #     from spacy.tokenizer import Tokenizer
        #     import en_core_web_sm
        #     nlp = en_core_web_sm.load()
        #     setattr(cls, 'q_tokenizer', Tokenizer(nlp.vocab))
        # return [t.text if '?' not in t.text else t.text[:-1] for t in cls.q_tokenizer(text.lower())]
        tokens = nltk.word_tokenize(text.lower())
        if tokens[-1] == '?':
            return tokens[:-1]
        return tokens

    @classmethod
    def a_tokenize(cls, answer):
        return answer

    @classmethod
    def load_combined_vocab(cls, data_dir='work_dir/data/CLEVR_v1.0'):
        data_dir = to_path(data_dir)
        q_vocab_file = data_dir.joinpath(f'{cls.name}/q_vocab.json')
        a_vocab_file = data_dir.joinpath(f'{cls.name}/a_vocab.json')
        if q_vocab_file.exists() and a_vocab_file.exists():
            q_vocab = Vocab.build_from_counter_file('q_vocab', data_dir.joinpath(cls.name), q_vocab_file,
                                                    special_symbols='<padding>')
            a_vocab = Vocab.build_from_counter_file('a_vocab', data_dir.joinpath(cls.name), a_vocab_file)
            return q_vocab, a_vocab

        ann_files = [data_dir.joinpath(f'questions/CLEVR_{split}_questions.json') for split in ('train', 'val')]
        q_anns = [json.load(a_file.open())['questions'] for a_file in ann_files]
        if not q_vocab_file.exists():
            questions = [ann_item['question'] for q_ann in q_anns for ann_item in q_ann]
            q_tokens = []
            for question in tqdm(questions, desc='Tokenizing questions'):
                q_tokens.append(cls.q_tokenize(question))
            q_vocab = Vocab.build_from_sequences('q_vocab', data_dir.joinpath(f'{cls.name}'), q_tokens,
                                                 special_symbols='<padding>')
            q_vocab.dump_to_file(q_vocab_file)
        else:
            q_vocab = Vocab.build_from_counter_file('q_vocab', data_dir.joinpath(cls.name), q_vocab_file,
                                                    special_symbols='<padding>')
        if not a_vocab_file.exists():
            answers = [ann_item['answer'] for ann in q_anns for ann_item in ann]
            a_tokens = []
            for answer in tqdm(answers, desc='Tokenizing answers'):
                a_tokens.append(cls.a_tokenize(answer))
            a_vocab = Vocab.build_from_sequences('a_vocab', data_dir.joinpath(f'{cls.name}'), a_tokens)
            a_vocab.dump_to_file(a_vocab_file)
        else:
            a_vocab = Vocab.build_from_counter_file('a_vocab', data_dir.joinpath(cls.name), a_vocab_file)
        q_vocab.strict = True
        a_vocab.strict = True
        return q_vocab, a_vocab

    @classmethod
    def load_combined_anns(cls, data_dir, data_split):
        data_dir = to_path(data_dir)
        ann_file = data_dir.joinpath(f'{cls.name}/{data_split}_ann.pt')
        if ann_file.exists():
            anns_t = torch.load(ann_file)
        else:
            sys_logger.info(f'Creating combined annotation files {ann_file.name}')
            q_vocab, a_vocab = cls.load_combined_vocab(data_dir)

            if data_split in ('valB_30k', 'valB_120k'):
                q_anns = json.load(data_dir.joinpath('questions', f'CLEVR_valB_questions.json').open())['questions']
                if data_split == 'valB_30k':
                    q_anns = q_anns[:30000]
                else:
                    q_anns = q_anns[30000:]
            else:
                q_anns = json.load(data_dir.joinpath('questions', f'CLEVR_{data_split}_questions.json').open())['questions']
            anns = defaultdict(list)
            for q_item in tqdm(q_anns, desc=f'Creating {ann_file.name}'):
                anns['q_ids'].append(q_item['question_index'])
                anns['img_ids'].append(q_item['image_index'])
                anns['img_names'].append(q_item['image_filename'])
                anns['q_tokens'].append(cls.q_tokenize(q_item['question']))
                anns['q_types'].append(q_item['question_family_index'])
                anns['a_tokens'].append(q_item['answer'])
                anns['q_lens'].append(len(anns['q_tokens'][-1]))
                q_label = [q_vocab[token] for token in anns['q_tokens'][-1]]
                q_label_t = torch.empty(47).fill_(q_vocab.padding_idx).long()
                q_label_t[:len(q_label)] = torch.tensor(q_label).long()
                anns['q_labels'].append(q_label_t)
                anns['a_labels'].append(a_vocab[q_item['answer']])
            anns_t = {name: to_tensor(value) for name, value in anns.items()}
            torch.save(anns_t, ann_file)
        return anns_t

    def build_fields(self):
        ann_reader = DictReader(self.load_combined_anns(self.data_dir, self.split))
        fields = list()
        fields.extend([ann_reader[f_name] for f_name in ('q_lens', 'a_labels', 'q_ids', 'q_labels')])
        img_idx_field = ann_reader['img_ids']

        if self.req_field_names is None or 'img_obj_feats' in self.req_field_names:
            feat_reader = H5Reader(self.data_dir.joinpath(self.name, f'clevr_original_{self.split}.h5'))
            feat_fields = [feat_reader[key] for key in ('img_obj_feats', 'img_obj_nums', 'img_obj_boxes', 'img_shapes')]

            def idx_map_fn(dset_idx):
                return img_idx_field[dset_idx].item()
            for field in feat_fields:
                field.idx_map_fn = idx_map_fn

            def depend_fn(x):
                if not isinstance(x, np.ndarray):
                    return torch.tensor(x)
                return torch.from_numpy(x)

            fields.extend(ProxyField(field.name, field, depend_fn=depend_fn) for field in feat_fields[:-1])

            def depend_fn(bb_locs, img_shape):
                bb_locs = torch.from_numpy(bb_locs)
                new_bb_locs = torch.empty_like(bb_locs)
                new_bb_locs[:, 0::2] = bb_locs[:, 0::2] / img_shape[0]
                new_bb_locs[:, 1::2] = bb_locs[:, 1::2] / img_shape[1]
                return new_bb_locs

            fields.append(ProxyField('img_obj_boxes', (feat_fields[-2], feat_fields[-1]), depend_fn))

        return fields

    def collate_fn(self, batch):
        if 'q_lens' in self.req_field_names:
            q_len_id = get_idx(self.req_field_names, 'q_lens')
            batch = sorted(batch, key=lambda x: x[q_len_id], reverse=True)
        batch_dict = super().collate_fn(batch)
        return batch_dict

    def __len__(self):
        return len(self['q_labels'])

    def evaluate(self, eval_log: dict):
        q_id2type_file = self.data_dir.joinpath(self.name, 'val_q_id2typ.json')
        if not q_id2type_file.exists():
            q_anns = json.load(self.data_dir.joinpath('questions/CLEVR_val_questions.json').open())['questions']
            q_id2type = {q_ann['question_index']: q_ann['program'][-1]['function'] for q_ann in q_anns}
            json.dump(q_id2type, q_id2type_file.open('w'))
        q_id2type = json.load(q_id2type_file.open())

        correct_by_q_type = defaultdict(list)
        for q_idx, is_true in eval_log.items():
            correct_by_q_type['Overall'].append(is_true)
            q_type = q_id2type[str(q_idx)]
            correct_by_q_type[q_type].append(is_true)
        acc_by_q_type = {}
        for q_type, vals in correct_by_q_type.items():
            vals = np.asarray(vals)
            # print(q_type, '%d / %d = %.2f' % (vals.sum(), vals.shape[0], 100.0 * vals.mean()))
            acc_by_q_type[q_type] = vals.mean()
        return acc_by_q_type










