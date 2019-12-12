#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from torchtext import data
from torchtext.data import Iterator, BucketIterator
import os
import re
import math
import torch
import numpy as np


def read_data(input_file):
    """Reads a data. """
    # <class 'list'>:
    # [[['军','队','政','治','人','民'],['B-n','I-n','B-n','I-n','B-n','I-n'],['P','S','P','S','S','P']],
    # [['偏', '重', '某', '些', '学', '科'],['B-v', 'I-v', 'B-r', 'I-r', 'B-n', 'I-n'],
    # ['P', 'S', 'S', 'S', 'S', 'P']],......]
    with open(input_file, encoding='utf8') as f:
        lines = []
        for line in f:
            labels = []
            pos = []
            character = []
            contends = line.strip()
            abbre = contends.strip().split(':')[0]
            abbre = list(abbre)
            full = contends.strip().split(':')[1]
            tokens = full.strip().split(' ')
            for token in tokens:
                token = token.strip().split('/')
                tok = list(token[0])
                lab = token[1]
                character = character + tok
                pos.append('B-' + lab)
                pos = pos + [('I-' + lab) for _ in range(len(tok) - 1)]
            if abbre == ['n']:
                labels = ['N' for _ in range((len(character)))]
            else:
                for cha in character:
                    if cha in abbre:
                        labels.append('P')
                    else:
                        labels.append('S')
            lines.append([character, pos, labels])
    return lines


class AbbDataset(data.Dataset):

    def __init__(self, char_field, pos_field, label_field, datafile, **kwargs):
        fields = [("char", char_field), ("pos", pos_field), ("label", label_field)]
        datas = read_data(datafile)
        examples = []
        for char, pos, label in datas:
            examples.append(data.Example.fromlist([char, pos, label], fields))
        super(AbbDataset, self).__init__(examples, fields, **kwargs)


def get_char_detail(train, other, embed_vocab=None):
    '''
        OOTV words are the ones do not appear in training set but in embedding vocabulary
        OOEV words are the ones do not appear in embedding vocabulary but in training set
        OOBV words are the ones do not appears in both the training and embedding vocabulary
        IV words the ones appears in both the training and embedding vocabulary
    '''

    char2type = {}  # type, 1:ootv, 2:ooev, 3:oobv, 4:iv
    ootv = 0
    ootv_set = set()
    ooev = 0
    oobv = 0
    iv = 0
    for sent in other:
        for w in sent:
            for c in w:
                if c not in char2type:
                    if c not in train.stoi:
                        if embed_vocab and (c in embed_vocab.stoi):
                            ootv += 1
                            ootv_set.add(c)
                            char2type[c] = 1
                        else:
                            oobv += 1
                            char2type[c] = 3
                    else:
                        if embed_vocab and (c in embed_vocab.stoi):
                            iv += 1
                            char2type[c] = 4
                        else:
                            ooev += 1
                            char2type[c] = 2
    print("IV {}\nOOTV {}\nOOEV {}\nOOBV {}\n".format(iv, ootv, ooev, oobv))
    return char2type, ootv_set


def extend(vocab, v, sort=False):
    words = sorted(v) if sort else v
    for w in words:
        if w not in vocab.stoi:
            vocab.itos.append(w)
            vocab.stoi[w] = len(vocab.itos) - 1


def unk_init(x):
    dim = x.size(-1)
    bias = math.sqrt(3.0 / dim)
    x.uniform_(-bias, bias)
    return x


def get_vectors(embed, vocab, pretrain_embed_vocab):
    oov = 0
    for i, word in enumerate(vocab.itos):
        index = pretrain_embed_vocab.stoi.get(word, None)  # digit or None
        if index is None:
            if word.lower() in pretrain_embed_vocab.stoi:
                index = pretrain_embed_vocab.stoi[word.lower()]
        if index:
            embed[i] = pretrain_embed_vocab.vectors[index]
        else:
            oov += 1
    print('train vocab oov %d \ntrain vocab + dev ootv + test ootv: %d' % (oov, len(vocab.stoi)))
    return embed


def get_entities(vocab, data, tag2id):  # (CHAR_TEXT.vocab, train_data, POS.vocab.stoi)
    entities = {}
    unk = 0
    conflict = 0
    for ex in data.examples:
        ens = get_chunks(ex.pos, tag2id, id_format=False)  # [('PER', 0, 1), ('PER', 2, 5)]
        for e in ens:
            entity_words = [ex.char[ix] if ex.char[ix] in vocab.stoi else vocab.UNK for ix in range(e[1], e[2])]
            entities.setdefault(' '.join(entity_words), set())
            entities[' '.join(entity_words)].add(e[0])
            if vocab.UNK in entity_words:
                unk += 1
            if len(entities[' '.join(entity_words)]) == 2:
                conflict += 1
    print("entities contains `UNK` %d\nconflict entities %d\nall entities: %d\n" % (unk, conflict, len(entities)))
    return entities  # {'lan yin yu': {'PER'}}


def load_iters(char_embed_size, char_vectors, pos_embedding_size, pos_vectors=None, batch_size=32, device="cpu",
               data_path='data'):
    # CHAR_NESTING = data.Field(tokenize=list)
    # CHAR_TEXT = data.NestedField(CHAR_NESTING)
    CHAR_TEXT = data.Field(batch_first=True, pad_token="<pad>", include_lengths=True)
    LABEL = data.Field(unk_token=None, pad_token="O", batch_first=True)
    POS = data.Field(unk_token=None, pad_token="O", batch_first=True)

    train_data = AbbDataset(CHAR_TEXT, POS, LABEL, os.path.join(data_path, "train_set.txt"))
    dev_data = AbbDataset(CHAR_TEXT, POS, LABEL, os.path.join(data_path, "dev_set.txt"))
    test_data = AbbDataset(CHAR_TEXT, POS, LABEL, os.path.join(data_path, "test_set.txt"))

    print("train token num / total char num: %d/%d" % (
        len(train_data.examples), np.array([len(_.char) for _ in train_data.examples]).sum()))
    print("dev token num / total char num: %d/%d" % (
        len(dev_data.examples), np.array([len(_.char) for _ in dev_data.examples]).sum()))
    print("test token num / total char num: %d/%d" % (
        len(test_data.examples), np.array([len(_.char) for _ in test_data.examples]).sum()))

    LABEL.build_vocab(train_data.label)
    CHAR_TEXT.build_vocab(train_data.char, max_size=50000, min_freq=1)
    POS.build_vocab(train_data.pos,dev_data.pos,test_data.pos)

    # ------------------- char oov analysis-----------------------
    print('*' * 50 + ' unique char details of dev set ' + '*' * 50)
    dev_char2type, dev_ootv_set = get_char_detail(CHAR_TEXT.vocab, dev_data.char, char_vectors)
    print('#' * 110)
    print('*' * 50 + ' unique char details of test set ' + '*' * 50)
    test_char2type, test_ootv_set = get_char_detail(CHAR_TEXT.vocab, test_data.word, char_vectors)
    print('#' * 110)
    CHAR_TEXT.vocab.dev_char2type = dev_char2type
    CHAR_TEXT.vocab.test_char2type = test_char2type

    # ------------------- extend char vocab with ootv chars -----------------------
    print('*' * 50 + 'extending ootv chars to vocab' + '*' * 50)
    ootv = list(dev_ootv_set.union(test_ootv_set))
    extend(CHAR_TEXT.vocab, ootv)
    print('extended %d chars' % len(ootv))
    print('#' * 110)

    # ------------------- generate char embedding -----------------------
    vectors_to_use = unk_init(torch.zeros((len(CHAR_TEXT.vocab), char_embed_size)))
    if char_vectors is not None:
        vectors_to_use = get_vectors(vectors_to_use, CHAR_TEXT.vocab, char_vectors)
    CHAR_TEXT.vocab.vectors = vectors_to_use

    print("char vocab size: ", len(CHAR_TEXT.vocab))
    print("pos vocab size: ", len(POS.vocab))
    print("label vocab size: ", len(LABEL.vocab))

    # ------------------- pos analysis-----------------------
    print('*' * 50 + ' get train entities ' + '*' * 50)
    train_entities = get_entities(CHAR_TEXT.vocab, train_data, POS.vocab.stoi)
    print('#' * 110)
    print('*' * 50 + ' get dev entities ' + '*' * 50)
    dev_entity2type = get_entities(CHAR_TEXT.vocab, dev_data, POS.vocab.stoi)
    print('#' * 110)
    print('*' * 50 + ' get test entities ' + '*' * 50)
    test_entity2type = get_entities(CHAR_TEXT.vocab, test_data, POS.vocab.stoi)
    print('#' * 110)
    CHAR_TEXT.vocab.dev_pos2type = dev_entity2type
    CHAR_TEXT.vocab.test_pos2type = test_entity2type

    # ------------------- generate pos embedding -----------------------
    vectors_to_use2 = unk_init(torch.zeros((len(POS.vocab), pos_embedding_size)))
    if char_vectors is not None:
        vectors_to_use2 = get_vectors(vectors_to_use2, POS.vocab, pos_vectors)
    POS.vocab.vectors = vectors_to_use2
    # ----------------------------------------------------------------------------------------

    train_iter = BucketIterator(train_data, batch_size=batch_size, device=device, sort_key=lambda x: len(x.char),
                                sort_within_batch=True, repeat=False, shuffle=True)
    dev_iter = Iterator(dev_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                        repeat=False, shuffle=False)
    test_iter = Iterator(test_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False, shuffle=False)
    return train_iter, dev_iter, test_iter, CHAR_TEXT, POS, LABEL


def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags, bioes=False, id_format=True):
    """
    Given a sequence of tags, group entities and their position
    """
    if not id_format:
        seq = [tags[_] for _ in seq]

    # We assume by default the tags lie outside a named entity
    default = tags["O"]

    idx_to_tag = {idx: tag for tag, idx in tags.items()}

    chunks = []

    chunk_class, chunk_type, chunk_start = None, None, None
    for i, tok in enumerate(seq):
        if tok == default and (chunk_class in (["E", "S"] if bioes else ["B", "I"])):
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_class, chunk_type, chunk_start = "O", None, None

        if tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                # Initialize chunk for each entity
                chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
            else:
                if bioes:
                    if chunk_class in ["E", "S"]:
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        if tok_chunk_class in ["B", "S"]:
                            chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
                        else:
                            chunk_class, chunk_type, chunk_start = None, None, None
                    elif tok_chunk_type == chunk_type and chunk_class in ["B", "I"]:
                        chunk_class = tok_chunk_class
                    else:
                        chunk_class, chunk_type = None, None
                else:  # BIO schema
                    if tok_chunk_class == "B":
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
                    # else:
                    #     chunk_class, chunk_type = None, None

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def test_get_chunks():
    print(get_chunks([1, 2, 2, 2, 1, 2],
                         {'O': 0, "B-n": 1, "I-n": 2}))
    print(get_chunks(["B-n", "I-n", "I-n", "I-n","B-nn", "B-rere", "I-rere"],
                     {'O': 0, "B-n": 1, "I-n": 2,"B-rere":3,"I-rere":4,"B-nn":5}, id_format=False))
