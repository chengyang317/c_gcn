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
import functools
import torch


logger = logging.getLogger(__name__)


__all__ = ['GraphVqa2Dataset', 'GraphVqa2CpDataset']


class GraphVqa2Dataset(Dataset):

    name = 'graph_vqa2'
    valid_field_names = ('q_ids', 'img_ids', 'q_tokens', 'img_names', 'img_shapes', 'a_tokens', 'a_counts', 'a_scores',
                         'q_lens', 'img_boxes', 'img_obj_feats', 'q_labels', 'a_label_scores', 'a_label_counts')

    def __init__(self,
                 data_dir: str = 'work_dir/data/vqa2',
                 split: str = 'train',
                 req_field_names: List[str] = ('img_obj_feats', 'q_labels', 'q_lens', 'a_label_scores', 'a_label_counts', 'q_ids'),
                 seq_len: int = 14,
                 batch_size: int = 64,
                 workers_num: int = 4,
                 shuffle: bool = None,
                 splits: List[str] = ('train', 'val'),
                 device=None,
                 ):
        self._question_vocab = None
        self._answer_vocab = None
        self.seq_len = seq_len
        if split == 'test':
            req_field_names = [item for item in req_field_names if item not in ('a_label_scores', 'a_label_counts')]
        super().__init__(data_dir, split, req_field_names, batch_size, workers_num, shuffle, splits, device)

    def build_fields(self):
        data_dir = self.data_dir.joinpath(self.name)
        ann_reader = DictReader(self.load_combined_anns(self.data_dir, self.split))
        img_id_field = ann_reader['img_ids']
        fields = list()
        fields.extend([ann_reader[name] for name in ('q_ids', 'q_tokens', 'img_ids', 'img_shapes', 'a_counts')])
        split_fields = ann_reader['splits']

        if self.req_field_names is None or 'img_obj_feats' in self.req_field_names:
            # add image object fields
            img_readers = {split: H5Reader(data_dir.joinpath(f'{split}_obj_feat_36.hdf5')) for split in self.split.split('_')}
            hd5_imgId2Idxs = {split: {img_id: idx for idx, img_id in enumerate(reader['img_ids'])} for split, reader in img_readers.items()}

            img_box_fields = {}
            bb_feature_fields = {}
            for split, img_reader in img_readers.items():
                def id_map_fn(dset_id):
                    img_id = img_id_field[dset_id]
                    return int(hd5_imgId2Idxs[split_fields[dset_id]][int(img_id)])

                for field in img_reader.fields:
                    field.idx_map_fn = id_map_fn

                def depend_fn(bb_locs, img_shape):
                    bb_locs = torch.from_numpy(bb_locs)
                    new_bb_locs = torch.empty_like(bb_locs)
                    new_bb_locs[:, 0::2] = bb_locs[:, 0::2] / img_shape[0]
                    new_bb_locs[:, 1::2] = bb_locs[:, 1::2] / img_shape[1]
                    return new_bb_locs

                img_box_field = ProxyField('img_boxes', (img_reader['img_boxes'], ann_reader['img_shapes']), depend_fn)
                img_box_fields[split] = img_box_field

                # add image obj features field

                def depend_fn(bb_feat, bb_locs):
                    bb_feat = torch.from_numpy(bb_feat)
                    return torch.cat((bb_feat, bb_locs), dim=-1)

                bb_feature_field = ProxyField('img_obj_feats', (img_reader['img_obj_feats'], img_box_field), depend_fn)
                bb_feature_fields[split] = bb_feature_field

            def switch_fn(idx, fields):
                split = split_fields[idx]
                return fields[split]

            fields.append(SwitchField('img_boxes', img_box_fields, switch_fn))
            fields.append(SwitchField('img_obj_feats', bb_feature_fields, switch_fn))

        if self.req_field_names is None or 'q_labels' in self.req_field_names:
            q_label_field = ProxyField('q_labels', ann_reader['q_labels'], depend_fn=lambda x: x[:self.seq_len])
            q_len_field = ProxyField('q_lens', ann_reader['q_lens'], depend_fn=lambda x: min(x, torch.tensor(self.seq_len)))
            fields.extend([q_label_field, q_len_field])

        if self.req_field_names is None or 'a_label_scores' in self.req_field_names and self.split != 'test':

            def depend_fn(a_scores):
                a_label_score = torch.empty(len(self.answer_vocab)).fill_(self.answer_vocab.padding_idx).float()
                for a_token, a_score in a_scores:
                    a_label_score[self.answer_vocab[a_token]] = a_score
                return a_label_score
            a_label_field = ProxyField('a_label_scores', ann_reader['a_scores'], depend_fn=depend_fn)
            fields.append(a_label_field)

        if self.req_field_names is None or 'a_label_counts' in self.req_field_names and self.split != 'test':

            def depend_fn(answers):
                a_label_count = torch.empty(len(self.answer_vocab)).fill_(self.answer_vocab.padding_idx).float()
                for a_token, a_count in answers:
                    a_label_count[self.answer_vocab[a_token]] = a_count
                return a_label_count
            a_count_field = ProxyField('a_label_counts', ann_reader['a_counts'], depend_fn=depend_fn)
            fields.append(a_count_field)

        if self.req_field_names is None or 'q_tokens' in self.req_field_names:
            fields.append(ann_reader['q_tokens'])
        if self.req_field_names is None or 'img_ids' in self.req_field_names:
            fields.append(ann_reader['img_ids'])
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
        if getattr(cls, 'q_tokenizer', None) is None:
            from spacy.tokenizer import Tokenizer
            import en_core_web_sm
            nlp = en_core_web_sm.load()
            setattr(cls, 'q_tokenizer', Tokenizer(nlp.vocab))
        return [t.text if '?' not in t.text else t.text[:-1] for t in cls.q_tokenizer(text.lower())]

    @classmethod
    def a_tokenize(cls, answer):
        if getattr(cls, 'a_tokenizer', None) is None:
            from .answer_preprocess import preprocess_answer
            setattr(cls, 'a_tokenizer', preprocess_answer)
        return cls.a_tokenizer(answer)

    @classmethod
    def load_combined_vocab(cls, data_dir='work_dir/data/vqa2'):
        data_dir = to_path(data_dir)
        q_vocab_file = data_dir.joinpath(f'{cls.name}/q_vocab.json')
        a_vocab_file = data_dir.joinpath(f'{cls.name}/a_vocab.json')
        if not q_vocab_file.exists():
            logger.info(f'Creating question vocab {q_vocab_file.name}')
            q_files = (f'v2_OpenEnded_mscoco_{key}_questions.json' for key in ('train2014', 'val2014', 'test2015'))
            q_files = (data_dir.joinpath(f'annotations/{q_file}') for q_file in q_files)
            questions = functools.reduce(lambda x, y: x+y, (json.load(q_file.open())['questions'] for q_file in q_files))
            q_tokens = []
            for q in tqdm.tqdm(questions):
                tokens = cls.q_tokenize(q['question'])
                q_tokens.append(tokens)
            q_vocab = Vocab.build_from_sequences('q_vocab', data_dir.joinpath(f'{cls.name}'), q_tokens, special_symbols='<padding>')
            q_vocab.dump_to_file(q_vocab_file)
        if not a_vocab_file.exists():
            logger.info(f'Creating answer vocab {a_vocab_file.name}')
            a_files = (f'v2_mscoco_{split}2014_annotations.json' for split in ('train', 'val'))
            a_files = (data_dir.joinpath(f'annotations/{a_file}') for a_file in a_files)
            answers = functools.reduce(lambda x, y: x+y, (json.load(a_file.open())['annotations'] for a_file in a_files))
            a_tokens = []
            for answer in tqdm.tqdm(answers):
                gtruth = cls.a_tokenize(answer['multiple_choice_answer'])
                a_tokens.append(gtruth)
            a_vocab = Vocab.build_from_sequences('a_vocab', data_dir.joinpath(f'{cls.name}'), a_tokens, max_word_num=3000, special_symbols='<padding>')
            a_vocab.dump_to_file(a_vocab_file)
        q_vocab = Vocab.build_from_counter_file('q_vocab', data_dir.joinpath(cls.name), q_vocab_file, special_symbols='<padding>')
        a_vocab = Vocab.build_from_counter_file('a_vocab', data_dir.joinpath(cls.name), a_vocab_file, special_symbols='<padding>')
        return q_vocab, a_vocab

    @classmethod
    def load_combined_anns(cls, data_dir: str='data/vqa2', data_split: str='train',):
        data_dir = to_path(data_dir)
        ann_file = data_dir.joinpath(f'{cls.name}/{data_split}_ann.pt')
        if ann_file.exists():
            anns_t = torch.load(ann_file)
            if 'splits' not in anns_t:
                anns_t['splits'] = [data_split] * len(anns_t['q_ids'])
                torch.save(anns_t, ann_file)
        else:
            logger.info(f'Creating combined annotation files {ann_file.name}')
            q_vocab, a_vocab = cls.load_combined_vocab(data_dir)
            if data_split in ('train', 'val'):
                origin_anns = json.load(data_dir.joinpath(f'annotations/v2_mscoco_{data_split}2014_annotations.json').open())['annotations']
                q_anns = json.load(data_dir.joinpath(f'annotations/v2_OpenEnded_mscoco_{data_split}2014_questions.json').open())['questions']
                ins_anns = json.load(data_dir.joinpath(f'annotations/instances_{data_split}2014.json').open())
                img_anns = {ann['id']: ann for ann in ins_anns['images']}
            elif data_split == 'test':
                origin_anns = None
                q_anns = json.load(data_dir.joinpath(f'annotations/v2_OpenEnded_mscoco_{data_split}2015_questions.json').open())['questions']
                ins_anns = json.load(data_dir.joinpath(f'annotations/image_info_{data_split}2015.json').open())
                img_anns = {ann['id']: ann for ann in ins_anns['images']}
            elif data_split == 'train_val':
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
            for idx, q in enumerate(tqdm.tqdm(q_anns)):
                anns['splits'].append(data_split)
                anns['q_ids'].append(q['question_id'])
                anns['img_ids'].append(q['image_id'])
                anns['q_tokens'].append(cls.q_tokenize(q['question']))
                anns['q_lens'].append(len(anns['q_tokens'][-1]))
                q_label = [q_vocab[token] for token in anns['q_tokens'][-1]]
                q_label_t = torch.empty(24).fill_(q_vocab.padding_idx).long()
                q_label_t[:len(q_label)] = torch.tensor(q_label).long()
                anns['q_labels'].append(q_label_t)

                img_ann = img_anns[q['image_id']]
                anns['img_names'].append(img_ann['file_name'])
                anns['img_shapes'].append((img_ann['width'], img_ann['height']))

                if origin_anns is None:
                    continue
                assert q['question_id'] == origin_anns[idx]['question_id']
                ori_ann = origin_anns[idx]
                anns['a_tokens'].append(cls.a_tokenize(ori_ann['multiple_choice_answer']))
                answers = []
                for ans in ori_ann['answers']:
                    answ = cls.a_tokenize(ans['answer'])
                    if answ in a_vocab.words:
                        answers.append(answ)
                anns['a_counts'].append(collections.Counter(answers).most_common())
                # a_label_count = torch.empty(len(a_vocab)).fill_(a_vocab.padding_idx).float()
                # for a_token, a_count in anns['a_counts'][-1]:
                #     a_label_count[a_vocab[a_token]] = a_count
                # anns['a_label_counts'].append(a_label_count)
                accepted_answers = sum([ans[1] for ans in anns['a_counts'][-1]])
                anns['a_scores'].append([(ans[0], ans[1]/accepted_answers) for ans in anns['a_counts'][-1]])
                # a_label_score = torch.empty(len(a_vocab)).fill_(a_vocab.padding_idx).float()
                # for a_token, a_score in anns['a_scores'][-1]:
                #     a_label_score[a_vocab[a_token]] = a_score
                # anns['a_label_scores'].append(a_label_score)
            anns_t = {key: to_tensor(value) for key, value in anns.items()}
            torch.save(anns_t, ann_file)
        return anns_t

    def __len__(self):
        return len(self['q_labels'])

    def collate_fn(self, batch):
        if 'q_lens' in self.req_field_names:
            q_len_id = get_idx(self.req_field_names, 'q_lens')
            batch = sorted(batch, key=lambda x: x[q_len_id], reverse=True)
        return super().collate_fn(batch)


class GraphVqa2CpDataset(Dataset):

    name = 'graph_vqa2_cp'
    valid_field_names = ('q_ids', 'img_ids', 'q_tokens', 'img_names', 'img_shapes', 'a_tokens', 'a_counts', 'a_scores',
                         'q_lens', 'img_boxes', 'img_obj_feats', 'q_labels', 'a_label_scores', 'a_label_counts')

    def __init__(self,
                 data_dir: str = 'work_dir/data/vqa2',
                 split: str = 'train',
                 req_field_names: List[str] = ('img_obj_feats', 'q_labels', 'q_lens', 'a_label_scores', 'a_label_counts'),
                 seq_len: int = 14,
                 ):
        self._question_vocab = None
        self._answer_vocab = None
        self.seq_len = seq_len
        super().__init__(data_dir, split, req_field_names)

    def build_fields(self):
        data_dir = self.data_dir.joinpath(self.name)
        ann_reader = DictReader(self.load_combined_anns(self.data_dir, self.split))
        img_id_field = ann_reader['img_ids']
        fields = list()
        fields.extend([ann_reader[name] for name in ('q_ids',)])
        split_fields = ann_reader['splits']

        if self.req_field_names is None or 'img_obj_feats' in self.req_field_names:
            # add image object fields
            img_readers = {split: H5Reader(data_dir.joinpath(f'{split}_obj_feat_36.hdf5')) for split in ('train', 'val')}
            hd5_imgId2Idxs = {split: {img_id: idx for idx, img_id in enumerate(reader['img_ids'])} for split, reader in img_readers.items()}

            img_box_fields = {}
            bb_feature_fields = {}
            for split, img_reader in img_readers.items():
                def id_map_fn(dset_id):
                    img_id = img_id_field[dset_id]
                    return int(hd5_imgId2Idxs[split_fields[dset_id]][int(img_id)])

                for field in img_reader.fields:
                    field.idx_map_fn = id_map_fn

                def depend_fn(bb_locs, img_shape):
                    bb_locs = torch.from_numpy(bb_locs)
                    new_bb_locs = torch.empty_like(bb_locs)
                    new_bb_locs[:, 0::2] = bb_locs[:, 0::2] / img_shape[0]
                    new_bb_locs[:, 1::2] = bb_locs[:, 1::2] / img_shape[1]
                    return new_bb_locs

                img_box_field = ProxyField('img_boxes', (img_reader['img_boxes'], ann_reader['img_shapes']), depend_fn)
                img_box_fields[split] = img_box_field

                # add image obj features field

                def depend_fn(bb_feat, bb_locs):
                    bb_feat = torch.from_numpy(bb_feat)
                    return torch.cat((bb_feat, bb_locs), dim=-1)

                bb_feature_field = ProxyField('img_obj_feats', (img_reader['img_obj_feats'], img_box_field), depend_fn)
                bb_feature_fields[split] = bb_feature_field

            def switch_fn(idx, fields):
                split = split_fields[idx]
                return fields[split]

            fields.append(SwitchField('img_boxes', img_box_fields, switch_fn))
            fields.append(SwitchField('img_obj_feats', bb_feature_fields, switch_fn))

        if self.req_field_names is None or 'q_labels' in self.req_field_names:
            q_label_field = ProxyField('q_labels', ann_reader['q_labels'], depend_fn=lambda x: x[:self.seq_len])
            q_len_field = ProxyField('q_lens', ann_reader['q_lens'], depend_fn=lambda x: min(x, torch.tensor(self.seq_len)))
            fields.extend([q_label_field, q_len_field])

        if self.req_field_names is None or 'a_label_scores' in self.req_field_names:

            def depend_fn(a_scores):
                a_label_score = torch.empty(len(self.answer_vocab)).fill_(self.answer_vocab.padding_idx).float()
                for a_token, a_score in a_scores:
                    a_label_score[self.answer_vocab[a_token]] = a_score
                return a_label_score
            a_label_field = ProxyField('a_label_scores', ann_reader['a_scores'], depend_fn=depend_fn)
            fields.append(a_label_field)

        if self.req_field_names is None or 'a_label_counts' in self.req_field_names:

            def depend_fn(answers):
                a_label_count = torch.empty(len(self.answer_vocab)).fill_(self.answer_vocab.padding_idx).float()
                for a_token, a_count in answers:
                    a_label_count[self.answer_vocab[a_token]] = a_count
                return a_label_count
            a_count_field = ProxyField('a_label_counts', ann_reader['a_counts'], depend_fn=depend_fn)
            fields.append(a_count_field)
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
        if getattr(cls, 'q_tokenizer', None) is None:
            from spacy.tokenizer import Tokenizer
            import en_core_web_sm
            nlp = en_core_web_sm.load()
            setattr(cls, 'q_tokenizer', Tokenizer(nlp.vocab))
        return [t.text if '?' not in t.text else t.text[:-1] for t in cls.q_tokenizer(text.lower())]

    @classmethod
    def a_tokenize(cls, answer):
        if getattr(cls, 'a_tokenizer', None) is None:
            from .answer_preprocess import preprocess_answer
            setattr(cls, 'a_tokenizer', preprocess_answer)
        return cls.a_tokenizer(answer)

    @classmethod
    def load_combined_vocab(cls, data_dir='work_dir/data/vqa2'):
        data_dir = to_path(data_dir)
        q_vocab_file = data_dir.joinpath(f'{cls.name}/q_vocab.json')
        a_vocab_file = data_dir.joinpath(f'{cls.name}/a_vocab.json')
        if not q_vocab_file.exists():
            logger.info(f'Creating question vocab {q_vocab_file.name}')
            q_files = (f'vqacp_v2_{key}_questions.json' for key in ('train', 'test'))
            q_files = (data_dir.joinpath(f'annotations/{q_file}') for q_file in q_files)
            questions = functools.reduce(lambda x, y: x+y, (json.load(q_file.open()) for q_file in q_files))
            q_tokens = []
            for q in tqdm.tqdm(questions):
                tokens = cls.q_tokenize(q['question'])
                q_tokens.append(tokens)
            q_vocab = Vocab.build_from_sequences('q_vocab', data_dir.joinpath(f'{cls.name}'), q_tokens, special_symbols='<padding>')
            q_vocab.dump_to_file(q_vocab_file)
        if not a_vocab_file.exists():
            logger.info(f'Creating answer vocab {a_vocab_file.name}')
            a_files = (f'vqacp_v2_{split}_annotations.json' for split in ('train', 'test'))
            a_files = (data_dir.joinpath(f'annotations/{a_file}') for a_file in a_files)
            answers = functools.reduce(lambda x, y: x+y, (json.load(a_file.open()) for a_file in a_files))
            a_tokens = []
            for answer in tqdm.tqdm(answers):
                gtruth = cls.a_tokenize(answer['multiple_choice_answer'])
                a_tokens.append(gtruth)
            a_vocab = Vocab.build_from_sequences('a_vocab', data_dir.joinpath(f'{cls.name}'), a_tokens, max_word_num=3000, special_symbols='<padding>')
            a_vocab.dump_to_file(a_vocab_file)
        q_vocab = Vocab.build_from_counter_file('q_vocab', data_dir.joinpath(cls.name), q_vocab_file, special_symbols='<padding>')
        a_vocab = Vocab.build_from_counter_file('a_vocab', data_dir.joinpath(cls.name), a_vocab_file, special_symbols='<padding>')
        return q_vocab, a_vocab

    @classmethod
    def load_combined_anns(cls, data_dir: str='data/vqa2', data_split: str='train',):
        data_dir = to_path(data_dir)
        ann_file = data_dir.joinpath(f'{cls.name}/{data_split}_ann.pt')
        if ann_file.exists():
            anns_t = torch.load(ann_file)
            if 'splits' not in anns_t:
                anns_t['splits'] = [data_split] * len(anns_t['q_ids'])
                torch.save(anns_t, ann_file)
        else:
            logger.info(f'Creating combined annotation files {ann_file.name}')
            q_vocab, a_vocab = cls.load_combined_vocab(data_dir)

            img_ann_dict = {}
            for split in ('train', 'val'):
                ins_anns = json.load(data_dir.joinpath(f'annotations/instances_{split}2014.json').open())
                img_anns = {ann['id']: ann for ann in ins_anns['images']}
                img_ann_dict[split] = img_anns

            if data_split in ('train', 'test'):
                origin_anns = json.load(data_dir.joinpath(f'annotations/vqacp_v2_{data_split}_annotations.json').open())
                q_anns = json.load(data_dir.joinpath(f'annotations/vqacp_v2_{data_split}_questions.json').open())

            else:
                raise NotImplementedError()

            anns = collections.defaultdict(list)
            for idx, q in enumerate(tqdm.tqdm(q_anns)):
                anns['splits'].append(q['coco_split'][:-4])
                anns['q_ids'].append(q['question_id'])
                anns['img_ids'].append(q['image_id'])
                anns['q_tokens'].append(cls.q_tokenize(q['question']))
                anns['q_lens'].append(len(anns['q_tokens'][-1]))
                q_label = [q_vocab[token] for token in anns['q_tokens'][-1]]
                q_label_t = torch.empty(24).fill_(q_vocab.padding_idx).long()
                q_label_t[:len(q_label)] = torch.tensor(q_label).long()
                anns['q_labels'].append(q_label_t)

                img_anns = img_ann_dict[anns['splits'][-1]]

                img_ann = img_anns[q['image_id']]
                anns['img_names'].append(img_ann['file_name'])
                anns['img_shapes'].append((img_ann['width'], img_ann['height']))

                if origin_anns is None:
                    continue
                assert q['question_id'] == origin_anns[idx]['question_id']
                ori_ann = origin_anns[idx]
                anns['a_tokens'].append(cls.a_tokenize(ori_ann['multiple_choice_answer']))
                answers = []
                for ans in ori_ann['answers']:
                    answ = cls.a_tokenize(ans['answer'])
                    if answ in a_vocab.words:
                        answers.append(answ)
                anns['a_counts'].append(collections.Counter(answers).most_common())

                accepted_answers = sum([ans[1] for ans in anns['a_counts'][-1]])
                anns['a_scores'].append([(ans[0], ans[1]/accepted_answers) for ans in anns['a_counts'][-1]])

            anns_t = {key: to_tensor(value) for key, value in anns.items()}
            torch.save(anns_t, ann_file)
        return anns_t

    def __len__(self):
        return len(self['q_labels'])

    def collate_fn(self, batch):
        if 'q_lens' in self.req_field_names:
            q_len_id = get_idx(self.req_field_names, 'q_lens')
            batch = sorted(batch, key=lambda x: x[q_len_id], reverse=True)
        return super().collate_fn(batch)



