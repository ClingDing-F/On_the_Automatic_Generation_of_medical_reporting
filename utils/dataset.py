#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/9 10:31
# @Author  : dreamlane
# @File    : dataset.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from Tag import Tag
import json
import numpy as np
import pandas as pd
from torchvision import transforms
from Vocabulary import Vocabulary
import pickle
import sys
sys.path.append("../")
from Vocabulary import *
from Tag import *


class ChestXrayDataSet(Dataset):
    def __init__(self, image_dir, file_list,caption_json, vocabulary,
                 s_max=10, n_max=50, transforms=None):
        self.image_dir = image_dir
        self.vocab = vocabulary
        self.caption=JsonReader(caption_json).data

        self.images, self.labels = self.__load_label_list(file_list)
        self.transform = transforms
        self.s_max = s_max
        self.n_max = n_max

    # def __generate_labels_list(self, file_list):
    #
    #     data = pd.read_csv(file_list)
    #     labels = []
    #     images = data['image_id']
    #     captions_to_label = data['caption_new']
    #     tag = Tag()
    #
    #     for item in captions_to_label:
    #         label = tag.tags2array(item)
    #         if sum(label) == 0:
    #             label[-1] = 1
    #         labels.append(label)
    #     caption=list(data['caption'].values)
    #     return images, caption, labels

    def __load_label_list(self, file_list):
        data = pd.read_csv(file_list)
        filename_list = list(data['image_id'])
        labels = []
        for labels_str in list(data['label']):
            labels_str_pre = labels_str[1:-1].replace(".", "").replace("\n", '')
            label = [int(i) for i in labels_str_pre.split()]
            labels.append(label)
        return filename_list, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]

        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        try:
            text = self.caption[image_name]
        except Exception as err:
            text = 'normal. '

        target = list()
        max_word_num = 0
        for i, sentence in enumerate(text.split('. ')):
            if i >= self.s_max:
                break
            sentence = sentence.split()
            if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max:
                continue
            tokens = list()
            tokens.append(self.vocab('<start>'))
            tokens.extend([self.vocab(token.strip()) for token in sentence])
            tokens.append(self.vocab('<end>'))
            if max_word_num < len(tokens):
                max_word_num = len(tokens)
            target.append(tokens)
        sentence_num = len(target)
        return image, image_name, list(label / np.sum(label)), target, sentence_num, max_word_num

def collate_fn(data):
    images, image_id, label, captions, sentence_num, max_word_num = zip(*data)
    images = torch.stack(images, 0)
    max_sentence_num = max(sentence_num)
    max_word_num = max(max_word_num)

    targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))

    prob = np.zeros((len(captions), max_sentence_num + 1))

    for i, caption in enumerate(captions):
        for j, sentence in enumerate(caption):
            targets[i, j, :len(sentence)] = sentence[:]
            prob[i][j] = len(sentence) > 0
    return images, image_id, torch.Tensor(label), targets, prob


def get_loader(image_dir, file_list, caption_json,vocabulary, transform,
               batch_size, s_max=10, n_max=50, shuffle=False):
    dataset = ChestXrayDataSet(image_dir=image_dir,
                               file_list=file_list,
                               caption_json=caption_json,
                               vocabulary=vocabulary,
                               s_max=s_max,
                               n_max=n_max,
                               transforms=transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn
                                              )
    return data_loader

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


if __name__ == '__main__':
        path = "/datasets/kf1d20/Indiana_data/"
        image_dir = path + "content/NLMCXR_png/"
        file_list = '../data/train.csv'

        caption_json='../data/captions.json'
        vocab_path = '../data/vocab.pkl'
        with open(vocab_path, 'rb') as f:
            vocab= pickle.load(f)

        batch_size = 6
        resize = 256
        crop_size = 224

        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        data_loader = get_loader(image_dir=image_dir,
                                 file_list=file_list,
                                 caption_json=caption_json,
                                 vocabulary=vocab, transform=transform,
                                 batch_size=batch_size,
                                 shuffle=False)

        for i, (image, image_id, label, target, prob) in enumerate(data_loader):
            # print(image.shape)
            print(i)
            print(image_id)
            # print(label)
            print(target)
            print(prob)

            print("*"*10)

