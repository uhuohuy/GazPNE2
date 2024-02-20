#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pdb
import sys
from collections import OrderedDict
import json
import math
import os
from pytorch_transformers import BertTokenizer
MODEL_PATH = 'bert-base-uncased'
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

import numpy as np

def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
        
#    with open(embeds_file) as json_data:
#         embeds_dict = json.load(json_data)
       
    return embeds_dict

def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

class BertEmbeds:
    def __init__(self, terms_file, embeds_file):
        self.terms_dict = read_terms(terms_file)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.embeddings = read_embeddings(embeds_file)
        
#        self.cosine_cache = {}
#        self.dist_threshold_cache = {}
#        self.normalize = normalize

    def get_embedding(self,text,tokenize=False):
        if (tokenize):
            tokenized_text = self.tokenizer.tokenize(text)
        else:
#            print('emb ',text)
            tokenized_text = text.split()
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        #print(text,indexed_tokens)
        vec =  self.get_vector(indexed_tokens)
        return vec
    
    def load_bert_embedding(self):
        embeds_cache = {}
        for key in self.terms_dict:
            vec = self.get_embedding(key)
            embeds_cache[key] = vec
        emb_dim = len(embeds_cache['the'])
        return embeds_cache, emb_dim


    def get_vector(self,indexed_tokens):
        vec = None
        if (len(indexed_tokens) == 0):
            return vec
        #pdb.set_trace()
        for i in range(len(indexed_tokens)):
            term_vec = self.embeddings[indexed_tokens[i]]
            if (vec is None):
                vec = np.zeros(len(term_vec))
            vec += term_vec
        sq_sum = 0
        for i in range(len(vec)):
            sq_sum += vec[i]*vec[i]
        sq_sum = math.sqrt(sq_sum)
        for i in range(len(vec)):
            vec[i] = vec[i]/sq_sum
        #sq_sum = 0
        #for i in range(len(vec)):
        #    sq_sum += vec[i]*vec[i]
        return vec
    
if __name__ == '__main__':
    BertEmbeds = BertEmbeds('data/uncased_vocab.txt', 'data/uncased_bert_vectors.txt')
    embeds, emb_dim = BertEmbeds.load_bert_embedding()
    print(emb_dim)
    print(embeds['flood'])
    print(embeds['flooded'])
