#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 17:51
# @Author  : dreamlane
# @File    : data_pre.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import cv2
from collections import Counter
import torch

from collections import defaultdict
import json
import pickle
from collections import Counter
import argparse
from utils.Vocabulary import *
from utils.Tag import *

import warnings
warnings.filterwarnings('ignore')




def save_file_list(file_name, data):
    with open(file_name, 'w') as output:
        for item in data:
            output.write(item)
            output.write('\n')


def generate_data(list_path, data_path, data, tags):
    num = 0
    with open(list_path, 'r') as input:
        with open(data_path, 'w') as output:
            for each in input.readlines():
                key = each.strip()
                value = data[key][-1]
                output.write(key)
                for tag in value:
                    if tag not in tags:
                        value.append('others')
                for tag in tags:
                    if tag in value:
                        num += 1
                        output.write(' 1')
                    else:
                        output.write(' 0')
                output.write('\n')
    print("{} file generated.".format(data_path))


def build_vocab(file_list_path, json_file, threshold):
    file_keys = []
    with open(file_list_path, 'r') as input:
        for each in input.readlines():
            file_keys.append(each[:-1])

    caption_reader = JsonReader(json_file)

    counter = Counter()

    for item in file_keys:

        if item in caption_reader.data.keys():
            text = caption_reader.data[item].replace('.', '').replace(',', '')
            counter.update(text.lower().split(' '))

    words = [word for word, cnt in counter.items() if cnt > 0 and word != '']
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    return vocab


class JsonReader(object):
    def __init__(self, json_file):
        self.data = self.__read_json(json_file)
        self.keys = list(self.data.keys())

    def __read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, item):
        return self.data[item]
        # return self.data[self.keys[item]]

    def __len__(self):
        return len(self.data)


def generate_id_tag_file(file_list_path, original_data, save_path):
    columns = ["image_id", "label"]
    df = pd.DataFrame(columns=columns)
    file_keys = []
    with open(file_list_path, 'r') as input:
        for each in input.readlines():
            file_keys.append(each[:-1])

    tags = Tag().static_tags
    for item in file_keys:
        value = original_data[item][-1]
        label = np.zeros(len(tags))
        for word in value:
            if word in tags:
                label[tags.index(word)] = int(1)
            if word not in tags:
                label[-1] = int(1)
        df = df.append(pd.Series([item, label], index=columns), ignore_index=True)

    df.to_csv(save_path, index=False)

def train_test_split(data):
    persons = list(data.keys())
    persons_train = persons[:5000]
    persons_cv = persons[5000:5500]
    persons_test = persons[5500:6000]
    return persons_train, persons_cv, persons_test

if __name__ == '__main__':
    f = open('./img2othersFull.pkl', 'rb')
    data = pickle.load(f)

    captions = {}
    for key in data:
        findings = data[key][0].replace('  ', ' ')
        discussion = data[key][1].replace('  ', ' ')
        caption = '. '.join([findings, discussion])
        caption = caption.replace(' .', '.').replace(',', '') \
            .replace('1', '<num>') \
            .replace('2', '<num>') \
            .replace('3', '<num>') \
            .replace('4', '<num>') \
            .replace('5', '<num>') \
            .replace('6', '<num>') \
            .replace('7', '<num>') \
            .replace('8', '<num>').replace('0', '<num>') \
            .replace('<num><num>', '<num>').replace('<num><num>', '<num>')
        if (len(discussion) != 0 and len(findings) != 0):
            captions[key] = caption
        # captions[key] = caption

    caption_path = './data/captions.json'
    with open(caption_path, 'w') as f:
        json.dump(captions, f)

    train_list, val_list, test_list = train_test_split(captions)

    train_list_path = "./data/train_list.txt"
    val_list_path = "./data/val_list.txt"
    test_list_path = "./data/test_list.txt"

    save_file_list(file_name=train_list_path, data=train_list)
    save_file_list(file_name=val_list_path, data=val_list)
    save_file_list(file_name=test_list_path, data=test_list)

    vocab_path = './data/vocab.pkl'
    vocab = build_vocab(file_list_path=train_list_path,
                        json_file=caption_path, threshold=0)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))

    train_data_save_path = "./data/train.csv"
    val_data_save_path = "./data/val.csv"
    test_data_save_path = "./data/test.csv"

    generate_id_tag_file(file_list_path=train_list_path,
                         original_data=data, save_path=train_data_save_path)
    generate_id_tag_file(file_list_path=val_list_path,
                         original_data=data, save_path=val_data_save_path)
    generate_id_tag_file(file_list_path=test_list_path,
                         original_data=data, save_path=test_data_save_path)