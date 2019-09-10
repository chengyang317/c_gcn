# coding=utf-8
import collections
import json
import numpy as np
from ..utils import to_path, to_seq
from typing import List, Dict
import torch


class Vocab(object):
    def __init__(self,
                 name=None,
                 data_dir=None,
                 idx2word: List[str]=None,
                 special_symbols=None,
                 word2idx: Dict[str, int]=None,
                 counter=None,
                 strict=True,
                 ):
        assert (counter is not None) ^ (idx2word is not None)
        self.name = name
        self.data_dir = to_path(data_dir)
        self.counter = counter
        self.special_symbols = special_symbols
        if counter is not None:
            self.idx2word = to_seq(special_symbols, []) + [word for word in list(self.counter)]
            self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        else:
            self.idx2word = idx2word
            self.word2idx = word2idx
        self.strict = strict

    @classmethod
    def build_from_sequences(cls, name, data_dir, sequences, min_count_num=None, max_word_num=None, special_symbols=None):
        counter = collections.Counter()
        for seq in sequences:
            seq = seq if isinstance(seq, (tuple, list)) else (seq, )
            counter.update(seq)
        most_common = counter.most_common(min_count_num)
        if max_word_num is not None:
            most_common = most_common[:max_word_num]
        counter = collections.Counter(dict(most_common))
        return cls(name, data_dir, counter=counter, special_symbols=special_symbols)

    def dump_to_file(self, json_file=None):
        json_file = json_file or self.data_dir.joinpath(self.name)
        json.dump(self.counter, json_file.open('w'))

    def save(self, json_file):
        json_file = json_file or self.data_dir.joinpath(self.name)
        json.dump({'idx2word': self.idx2word, 'word2idx': self.word2idx}, json_file.open('w'))

    @classmethod
    def from_file(cls, name, data_dir, json_file, strict=False):
        json_dict = json.load(json_file.open())
        return cls.build_from_statistic(json_dict['idx2word'], json_dict['word2idx'], name, data_dir, strict)

    @classmethod
    def build_from_counter_file(cls, name, data_dir, json_file, special_symbols=None):
        counter = collections.Counter(json.load(json_file.open()))
        return cls(name, data_dir, counter=counter, special_symbols=special_symbols)

    @classmethod
    def build_from_statistic(cls, idx2word, word2idx, name=None, data_dir=None, strict=False):
        return cls(name, data_dir, idx2word=idx2word, word2idx=word2idx, strict=strict)

    def __len__(self):
        return len(self.idx2word)

    @property
    def words(self):
        return self.idx2word

    @property
    def padding_idx(self):
        return self.word2idx['<padding>']

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.idx2word[item]
        elif isinstance(item, str):
            if item not in self.word2idx:
                if self.strict:
                    raise IndexError(f'Cannot find {item} in vocab')
                else:
                    if '<padding>' not in self.word2idx:
                        raise IndexError(f'Cannot find {item} in vocab')
                    else:
                        item = '<padding>'
            return self.word2idx[item]
        else:
            raise NotImplementedError()

    def glove_embed(self, glove_name):
        if getattr(self, '_glove_embeds', None) is None:
            self._glove_embeds = {}
        if glove_name not in self._glove_embeds:
            glove_embed_file = self.data_dir.joinpath(f'{self.name}_{glove_name}.pt')
            if not glove_embed_file.exists():
                glove_origin_file = self.data_dir.joinpath(f'{glove_name}.txt')
                glove_embeds = {}
                for line in glove_origin_file.open():
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype=np.float32)
                    glove_embeds[word] = coefs
                emb_dim = len(next(iter(glove_embeds.values())))
                q_embed = torch.zeros(len(self.idx2word), emb_dim).float()
                for word, word_idx in self.word2idx.items():
                    embed = glove_embeds.get(word, None)
                    if embed is not None:
                        q_embed[word_idx] = torch.from_numpy(embed)
                torch.save(q_embed, glove_embed_file)
            else:
                q_embed = torch.load(glove_embed_file)
            self._glove_embeds[glove_name] = q_embed
        return self._glove_embeds[glove_name]