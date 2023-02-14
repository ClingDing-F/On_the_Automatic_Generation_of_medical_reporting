#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/10 14:29
# @Author  : dreamlane
# @File    : test.py
# @Software: PyCharm
import time
import pickle
import time
import pickle
import argparse
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from PIL import Image
import cv2
import json
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
sys.path.append("./utils/")
from utils.model import *
from utils.dataset import *
from utils.loss import *
from utils.logger import Logger


class CaptionSampler(object):
    def __init__(self, args):
        self.args = args

        self.vocab = self.__init_vocab()
        self.tagger = self.__init_tagger()
        self.transform = self.__init_transform()
        self.data_loader = self.__init_data_loader(self.args.file_list, self.transform)
        self.model_state_dict = self.__load_mode_state_dict()

        self.extractor = self.__init_visual_extractor()
        self.mlc = self.__init_mlc()
        self.co_attention = self.__init_co_attention()
        self.sentence_model = self.__init_sentence_model()
        self.word_model = self.__init_word_word()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()
        self.s = []

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        print("Vocab Size:{}\n".format(len(vocab)))
        print("Init_vocab:success", self.args.vocab_path)
        return vocab

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def __init_tagger(self):
        return Tag()

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __init_data_loader(self, file_list, transform):
        data_loader = get_loader(image_dir=self.args.path + self.args.image_dir,
                                 file_list=file_list,
                                 vocabulary=self.vocab, transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        print("_init_data_loader ", file_list)
        return data_loader

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(os.path.join(self.args.model_dir, self.args.load_model_path))
            print("[Load Model-{} Succeed!]".format(self.args.load_model_path))
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)

        if self.model_state_dict is not None:
            print("Visual Extractor Loaded!")
            model.load_state_dict(self.model_state_dict['extractor'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.out_features,
                    k=self.args.k)

        if self.model_state_dict is not None:
            print("MLC Loaded!")
            model.load_state_dict(self.model_state_dict['mlc'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_co_attention(self):
        model = CoAttention(version=self.args.attention_version,
                            embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        if self.model_state_dict is not None:
            print("Co-Attention Loaded!")
            model.load_state_dict(self.model_state_dict['co_attention'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_sentence_model(self):
        model = SentenceLSTM(version=self.args.sent_version,
                             embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers,
                             dropout=self.args.dropout,
                             momentum=self.args.momentum)

        if self.model_state_dict is not None:
            print("Sentence Model Loaded!")
            model.load_state_dict(self.model_state_dict['sentence_model'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_word_word(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers,
                         n_max=self.args.n_max)

        if self.model_state_dict is not None:
            print("Word Model Loaded!")
            model.load_state_dict(self.model_state_dict['word_model'])

        if self.args.cuda:
            model = model.cuda()

        return model

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def __to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __save_json(self, result):
        result_path = os.path.join(self.args.model_dir, self.args.result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

    def generate(self):
        self.extractor.eval()
        self.mlc.eval()
        self.co_attention.eval()
        self.sentence_model.eval()
        self.word_model.eval()

        progress_bar = tqdm(self.data_loader, desc='Generating')
        results = {}

        for images, image_id, label, captions, _ in progress_bar:
            images = self.__to_var(images, requires_grad=False)
            visual_features, avg_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)

            sentence_states = None
            prev_hidden_states = self.__to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))
            pred_sentences = {}
            real_sentences = {}
            for i in image_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}

            for i in range(self.args.s_max):
                ctx, alpha_v, alpha_a = self.co_attention.forward(avg_features, semantic_features, prev_hidden_states)

                topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
                                                                                           prev_hidden_states,
                                                                                           sentence_states)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)

                start_tokens = np.zeros((topic.shape[0], 1))
                start_tokens[:, 0] = self.vocab('<start>')
                start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)

                sampled_ids = self.word_model.sample(topic, start_tokens)
                prev_hidden_states = hidden_state

                sampled_ids = torch.Tensor(sampled_ids) * p_stop.cpu()

                # self._generate_cam(image_id, visual_features, alpha_v, i)

                for id, array in zip(image_id, sampled_ids):
                    pred_sentences[id][i] = self.__vec2sent(array.cpu().detach().numpy())

            for id, array in zip(image_id, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)

            for id, pred_tag, real_tag in zip(image_id, tags, label):
                results[id] = {
                    'Real Tags': self.tagger.inv_tags2array(real_tag),
                    'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy()),
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }

        self.__save_json(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    # Path Argument
    parser.add_argument('--model_dir', type=str, default='./report_models/v4/20230211-11:18')
    parser.add_argument('--path', type=str, default="/datasets/kf1d20/Indiana_data/",
                    help='the path for data')
    parser.add_argument('--image_dir', type=str, default="content/NLMCXR_png/",
                    help='the path for image')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_train.pkl',
                    help='the path for vocabulary object')
    parser.add_argument('--file_list', type=str, default='./data/test.csv',
                    help='the path for test file list')
    parser.add_argument('--load_model_path', type=str, default='train_best_loss.pth.tar',
                    help='The path of loaded model')

    # transforms argument
    parser.add_argument('--resize', type=int, default=224,
                    help='size for resizing images')

    # CAM
    parser.add_argument('--cam_size', type=int, default=224)
    parser.add_argument('--generate_dir', type=str, default='cam')

    # Saved result
    parser.add_argument('--result_path', type=str, default='results',
                    help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='v4',
                    help='the name of results')

    """
    Model argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='densenet201',
                    help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=False,
                    help='not using pretrained model when training')

    # MLC
    parser.add_argument('--classes', type=int, default=211)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v1')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v1')
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)

    """
    Generating Argument
    """
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    parser.add_argument('--batch_size', type=int, default=16)

    # Loss function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args([])
    args.cuda = torch.cuda.is_available()

    torch.cuda.set_device(2)
    sampler = CaptionSampler(args)
    sampler.generate()
