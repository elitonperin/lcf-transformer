# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

from transformers import BertModel
from transformers import BertTokenizer
from transformers.modeling_bert import BertPooler, BertSelfAttention
from dotmap import DotMap
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import copy
import numpy as np
import logging
import argparse
import math
import sys
import os
import pandas as pd
from time import strftime, localtime
import random
import pickle
from tqdm.notebook import tqdm
from time import time

# The code is based on repository: https://github.com/yangheng95/LCF-ABSA

class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len), 
                                  dtype=torch.float32,
                                  device=self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class LCF_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert   # Default to use single Bert and reduce memory requirements
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.linear_single = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    # def feature_dynamic_mask(self, text_local_indices, aspect_indices):
    #     texts = text_local_indices#.cpu().numpy()
    #     asps = aspect_indices#.cpu().numpy()
    #     mask_len = self.opt.SRD
    #     masked_text_raw_indices = torch.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim), device=self.opt.device)
    #     for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
    #         # asp_len = torch.count_nonzero(asps[asp_i]) - 2
    #         asp_len = torch.sum(asps[asp_i] != 0) - 2
    #         try:
    #             #asp_begin = torch.argwhere(texts[text_i] == asps[asp_i][1], device=self.opt.device)[0][0]
    #             asp_begin = torch.where(texts[text_i] == asps[asp_i][1], device=self.opt.device)[0][0]
    #         except:
    #             continue
    #         if asp_begin >= mask_len:
    #             mask_begin = asp_begin - mask_len
    #         else:
    #             mask_begin = 0
    #         for i in range(mask_begin):
    #             masked_text_raw_indices[text_i][i] = torch.zeros((self.opt.bert_dim), device=self.opt.device)
    #         for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
    #             masked_text_raw_indices[text_i][j] = torch.zeros((self.opt.bert_dim), device=self.opt.device)
    #     # masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
    #     return masked_text_raw_indices#.to(self.opt.device)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices#.cpu().numpy()
        asps = aspect_indices#.cpu().numpy()
        mask_len = self.opt.SRD
        # masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
        #                                   dtype=np.float32)
        masked_text_raw_indices = torch.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim), device=self.opt.device)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            # asp_len = np.count_nonzero(asps[asp_i]) - 2
            # asp_len = torch.count_nonzero(asps[asp_i]) - 2  # torch 1.8
            asp_len = torch.sum(asps[asp_i] != 0) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = torch.zeros((self.opt.bert_dim), dtype=torch.float)
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = torch.zeros((self.opt.bert_dim), dtype=torch.float)
        if type(masked_text_raw_indices) == np.ndarray:
          masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices).to(self.opt.device)
        return masked_text_raw_indices#.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices#.cpu().numpy()
        asps = aspect_indices#.cpu().numpy()
        masked_text_raw_indices = torch.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim))
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = torch.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = torch.zeros(np.count_nonzero(texts[text_i]))
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2
                                        - self.opt.SRD)/np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    # def get_embeddings(self, inputs):
    #   return 


    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]

        bert_spc_out, _ = self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)

        bert_local_out, _ = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)

        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
        mean_pool = self.linear_double(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)

        return dense_out

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class LCF2Encoder():
  def __init__(self, model=None, path_model=None, 
               tokenizer=None, device='cpu', max_seq_len=80,
               pretrained_bert_name='bert-base-cased', profiling=False, embeddings_type='CLS', eval_mode=False):
    self.device = device
    self.max_seq_len = max_seq_len
    self.pretrained_bert_name = pretrained_bert_name
    self.tokenizer = tokenizer
    self.model = model
    self.embeddings_type = embeddings_type
    self.profiling = profiling
    self.eval_mode = eval_mode

    if self.tokenizer is None:
      self.tokenizer = Tokenizer4Bert(self.max_seq_len, pretrained_bert_name)

    if self.model is None:
      if path_model is None:
        self.model = self.load_our_model()
      else:
        self.model = self.load_our_model(path_model)
        
        
  def load_our_model(self, 
                     path_model=None):
    # opt = config_opt(opt, log_config)
    self.opt = DotMap()
    self.opt.max_seq_len = self.max_seq_len
    self.opt.pretrained_bert_name = self.pretrained_bert_name
    self.opt.local_context_focus = 'cdm'
    self.opt.bert_dim = 768
    self.opt.dropout = 0.1
    self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    self.opt.polarities_dim = 3
    self.opt.SDR = 3
    bert = BertModel.from_pretrained(self.pretrained_bert_name)
    model = LCF_BERT(bert, self.opt)
    
    if path_model is None:
      print('try !gdown https://drive.google.com/uc?id=1gMcCF7e8DvfssLOsmOlS28pIFn61ewW5')
      print("pass with parameter: lcf = LCF2Encoder(path_model='model_cpu_best.pt')")
      raise NameError('Model not found.')

    m = torch.load(path_model, 
                   map_location=self.opt.device)
    model.load_state_dict(m)
    model = model.to(self.opt.device)
    return model

  def local(self, text_bert_indices):
    bert_local_out, _ = self.model.bert_local(text_bert_indices)
    if self.embeddings_type == 'CLS':
      bert_local_out = [ bert_sample[0] for bert_sample in bert_local_out ]
    return bert_local_out

  def spc(self, text_bert_indices, bert_segments_ids):
    bert_spc_out, _ = self.model.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids)
    if self.embeddings_type == 'CLS':
      bert_spc_out = [ bert_sample[0] for bert_sample in bert_spc_out ]
    return bert_spc_out

  def local_spc_attention(self, texts_raw_bert_indices, bert_segments_ids, 
                          text_bert_indices, aspect_indices):
    
    # texts_raw_bert_indices, bert_segments_ids, text_bert_indices, aspect_indices = self.process_texts_with_aspects(texts, aspects)
    t0 = time()
    bert_local_out, _ = self.model.bert_local(texts_raw_bert_indices)
    if self.profiling: print('bert_local', time()-t0)
    t0 = time()
    bert_spc_out, _ = self.model.bert_spc(text_bert_indices,
                                          token_type_ids=bert_segments_ids)
    if self.profiling: print('bert_spc', time()-t0)
    t0 = time()

    if self.opt.local_context_focus == 'cdm':
        masked_local_text_vec = self.model.feature_dynamic_mask(texts_raw_bert_indices, aspect_indices)
        if self.profiling: print('feature_dynamic_mask', time()-t0)
        t0 = time()
        bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

    elif self.opt.local_context_focus == 'cdw':
        weighted_text_local_features = self.model.feature_dynamic_weighted(texts_raw_bert_indices, aspect_indices)
        if self.profiling: print('feature_dynamic_weighted', time()-t0)
        t0 = time()
        bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

    if self.profiling: print('mul', time()-t0)
    t0 = time()
    out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
    if self.profiling: print('cat', time()-t0)
    t0 = time()
    mean_pool = self.model.linear_double(out_cat)
    if self.profiling: print('linear_double', time()-t0)
    t0 = time()
    self_attention_out = self.model.bert_SA(mean_pool)
    if self.profiling: print('bert_SA', time()-t0)
    t0 = time()
    if self.embeddings_type == 'CLS':
      self_attention_out = [ bert_sample[0] for bert_sample in self_attention_out ]
    return self_attention_out

  def encode(self, texts, aspects=None, mode='local', batch=500, profiling=False):
    
    # fazer um laço com processamento de texto já feito OK
    # transformer usa 1000 de batch (512 dim. destilbert) 
    
    # validação das embeddings:
    # lcf vs destilbert:
    # sent. + asp => emb.  fazer a classificação com MLP
    # sent. => emb. fazer a classificação com MLP

    texts_raw_bert_indices, bert_segments_ids, text_bert_indices, aspect_indices = self.process_texts_with_aspects(texts, aspects)

    texts_raw_bert_indices = texts_raw_bert_indices.to(self.opt.device)
    bert_segments_ids = bert_segments_ids.to(self.opt.device)
    text_bert_indices = text_bert_indices.to(self.opt.device)
    aspect_indices = aspect_indices.to(self.opt.device)

    if self.eval_mode:
      self.model.eval()
      self.model.bert_spc.eval()
      self.model.bert_local.eval()
      self.model.bert_SA.eval()
    with torch.no_grad():
      L = []
      for i in tqdm(range(0, len(texts), batch)):
        outs = None
        if mode == 'local':
          outs = self.local(text_bert_indices[i:i+batch])
        elif mode == 'spc':
          outs = self.spc(text_bert_indices[i:i+batch], bert_segments_ids[i:i+batch])        
        elif mode == 'local+spc':
          outs = self.local_spc_attention(texts_raw_bert_indices[i:i+batch], 
                                          bert_segments_ids[i:i+batch], 
                                          text_bert_indices[i:i+batch], 
                                          aspect_indices[i:i+batch])
        elif mode == 'lcf_bert':
          embeddings_type = self.embeddings_type
          self.embeddings_type = None
          self_attention_out = self.local_spc_attention(texts_raw_bert_indices[i:i+batch], 
                                                        bert_segments_ids[i:i+batch], 
                                                        text_bert_indices[i:i+batch], 
                                                        aspect_indices[i:i+batch])
          self.embeddings_type = embeddings_type
          outs = self.model.bert_pooler(self_attention_out)
        else:
          raise 'Not configured for ' + mode
        t0 = time()
        L += outs
        # L += [ bert_sample.cpu().numpy()[0] for bert_sample in outs ]
        # L += [ bert_sample[0] for bert_sample in outs ]
        if self.profiling: print('List', time()-t0)

    L = [item.cpu().numpy() for item in L]
    return np.array(L)
  
  def get_aspect_embeddings():
    ''''TO DO'''
    pass
  
  def get_mean_embeddings():
    ''''TO DO'''
    pass

  def process_texts(self, texts):
    texts_raw_bert_indices = [ self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]") for text in texts]
    return torch.Tensor(texts_raw_bert_indices).to(torch.int64)

  def process_texts_with_aspects(self, texts, aspects):
    L_texts_raw_bert_indices = []
    L_bert_segments_ids = []
    L_text_bert_indices = []

    for text, aspect in tqdm(zip(texts, aspects), total=len(texts)):
      text_raw = self.tokenizer.text_to_sequence(text)
      texts_raw_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
      aspect_indices = self.tokenizer.text_to_sequence(aspect)
      aspect_len = np.sum(aspect_indices != 0)
      text_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")
      bert_segments_ids = np.asarray([0] * (np.sum(text_raw != 0) + 2) + [1] * (aspect_len + 1))
      bert_segments_ids = pad_and_truncate(bert_segments_ids, self.tokenizer.max_seq_len)
      L_texts_raw_bert_indices.append(texts_raw_bert_indices)
      L_bert_segments_ids.append(bert_segments_ids)
      L_text_bert_indices.append(text_bert_indices)

    L_texts_raw_bert_indices = torch.Tensor(L_texts_raw_bert_indices).to(torch.int64)
    L_bert_segments_ids = torch.Tensor(L_bert_segments_ids).to(torch.int64)
    L_text_bert_indices = torch.Tensor(L_text_bert_indices).to(torch.int64)
    aspect_indices = torch.Tensor(aspect_indices).to(torch.int64)
    return L_texts_raw_bert_indices, L_bert_segments_ids, L_text_bert_indices, aspect_indices

