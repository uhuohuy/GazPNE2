#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:03:40 2020

@author: hu_xk
"""
import os
import sys
from datetime import datetime
from place_tagger import place_tagging
from utility import *
import argparse
import torch
from Model import C_LSTM

#sys.path.append(os.path.abspath('unsupervised_NER'))
import time
# from main_NER import UnsupNER
import SentWrapper
MASK_TAG = "entity"
DISPATCH_MASK_TAG = "entity"
MODEL_PATH ='bert-large-cased'



def read_ent_file(ent_file):
    file1 = open(ent_file, 'r')
    Lines = file1.readlines()
    word_ent_dict = {}
    # Strips the newline character
    for line in Lines:
        words = re.split('\s+', line)
        cur_ent = {}
        i = 1
        while i < len(words)-1:
            cur_ent[words[i]]=float(words[i+1])
            i += 2
        word_ent_dict[words[0]] = cur_ent
    ent_dict_adv = {}
    for key in word_ent_dict.keys():
        ent_dict_adv[key] = word_ent_dict[key]
        if key.lower() not in word_ent_dict.keys():
            ent_dict_adv[key.lower()] = word_ent_dict[key]
    return ent_dict_adv


class GazPNE2:
    def __init__(self,model_ID='0622143005',epoch=4, hidden=150, filter_l=1, osm = 1, osmembed=2, bool_general_check=1,general_words_num=26000, emb=1, bool_mb_gaze=0, neigh=301, context_neigh=301, weight=4, context_model=0, intrinsci_model=0):
        self.desc_singleton = SentWrapper.SentWrapper(context_model)
#        self.desc_singleton.descs = SentWrapper.read_descs(DESC_FILE_ADV)
#        self.cluster_singleton = dist_v2.BertEmbeds('cache/',0,'data/vocab.txt','data/bert_vectors.txt',True,True,'data/labels.txt','data/stats_dict.txt','data/preserve_1_2_grams.txt','data/glue_words.txt')
#        self.common_descs = read_common_descs(cf.read_config()["COMMON_DESCS_FILE"])
        self.Word_Entities = read_ent_file('data/word_ent'+str(neigh)+'w'+str(weight)+'.txt')   # word_ent100_pure.txt
        self.Word_Entities_C = self.Word_Entities #dist_v2.read_ent_file('data/word_ent'+str(context_neigh)+'w'+str(weight)+'.txt')   # word_ent100_pure.txt
        
        self.model_ID = model_ID
        fc_file='data/fc.txt'
        self.fc_tokens = extract_tokens(fc_file)
        file_name = 'data/osm_abbreviations_globe.csv'
        sim_abv = abbrevison1(file_name)
        self.abv_punk = {}
        for key in sim_abv.keys():
            new_abv = ''
            for i, char in enumerate(key):
                    new_abv += char + '.'
            self.abv_punk[new_abv]=key
        #load the osm place names
        
        start_time = time.time()
    
        if osm:
    #        if args.F1 == 4:
            osm_names = load_osm_names_fre1('data/'+str(model_ID)+str(epoch)+'.txt', [], aug_count = 1)
            osm_names = [item for item in osm_names if len(item) > 1]
            self.osm_names = set(osm_names)
    #        else:
    #            osm_names = load_osm_names_fre('data/country.txt', [], aug_count = 1)
    #            osm_names = set(osm_names)
        else:
            self.osm_names = []
        # print('osm_names', time.time()-start_time)
        start_time = time.time()
            
        if bool_general_check:
            general_file = 'data/general_place'+str(general_words_num)+'.txt'
            if os.path.isfile(general_file):
                general_place_list = load_osm_names_fre(general_file, [], aug_count = 1)
            else:
                general_place_list = []
            # candidate_file = 'data/candidates'+str(args.general_words)+'.txt'
            # if os.path.isfile(candidate_file):
            #     candidate_words = load_osm_names_fre(candidate_file, [], aug_count = 1)
            #     general_place_list.extend(candidate_words)
        else:
            general_place_list = []

        self.PAD_idx = 0
        self.s_max_len = 10
        bigram_file = 'model/'+model_ID+'-bigram.txt'
        # hcfeat_file = 'model/'+model_ID+'-hcfeat.txt'
        self.START_WORD = 'hhh' #'start_string_taghu' # 'hhh'
        self.bigram_model = load_bigram_model(bigram_file)
        # bigram_model = {}
        self.bool_mb_gaze = bool_mb_gaze
        if bool_mb_gaze:
            gazetteer_emb_file = 'data/osm_vector'+str(osmembed)+'.txt'
            self.gazetteer_emb,self.gaz_emb_dim = load_embeding(gazetteer_emb_file)
        else:
            self.gazetteer_emb = []
            self.gaz_emb_dim = 0
        # general_words = list(set(general_words).difference(set(category_words_tuple)))
        self.general_words = general_place_list
        self.general_words.extend(['0','00','000','0000'])
        self.general_words = list(set(self.general_words).difference(set([tuple(['s'])])))
        # print(general_words)
        file_name = 'data/osm_abbreviations_globe.csv'
        self.abbr_dict = abbrevison1(file_name)
        start_time = time.time()
    
        # char_hc_emb,_ = load_embeding(hcfeat_file)
        self.char_hc_emb = {}
        # if bool_debug:
        #     print('hcfeat_file', time.time()-start_time)
        # start_time = time.time()
    
        word_idx_file = 'model/'+model_ID+'-vocab.txt'
        self.word2idx, self.max_char_len = load_word_index(word_idx_file)
        # if bool_debug:
        #     print('load_word_index', time.time()-start_time)
        # start_time = time.time()
    
        self.max_char_len = 20
        if emb==4:
            self.glove_emb = {}    
            self.emb_dim = 1024
        else:
            self.glove_emb = {}    
            self.emb_dim = 50
    
        self.weight_l = self.emb_dim+self.gaz_emb_dim+6
        weights_matrix = np.zeros((len(self.word2idx.keys()), self.weight_l))
        self.weights_matrix= torch.from_numpy(weights_matrix)
        tag_to_ix = {"p": 0, "n": 1}
        self.HIDDEN_DIM = hidden
        self.lstm_dim = hidden
        model_path = 'model/'+model_ID+'epoch'+str(epoch)+'.pkl'
        self.DROPOUT = 0.5
        self.flex_feat_len = 3
        self.fileter_l = filter_l
        start_time = time.time()
    
        self.model = C_LSTM(self.weights_matrix, self.HIDDEN_DIM, self.fileter_l, self.lstm_dim, len(tag_to_ix), self.flex_feat_len, self.DROPOUT)
        self.model.load_state_dict(torch.load(model_path,map_location='cpu'))
        self.model.eval()
        # if bool_debug:
        #     print('load_state_dict', time.time()-start_time)
        self.np_word_embeds = self.model.embedding.weight.detach().numpy() 

    def context_cue_new(self, masked_sent, bool_tweet_bert=0, ent=[],  ori_masked_sen = '', bool_debug=0, bool_formal=0):
        ent_prob = {}
        ent_prob['LOC'] = 0
        ent_prob_gen = {}
        ent_prob_gen['LOC'] = 0
        descs = []
        if bool_tweet_bert:
#            start_time = time.time()
            if not bool_formal:
                descs, desc_probs = self.desc_singleton.punct_sentence_tweet(masked_sent)
            else:
                descs, desc_probs = self.desc_singleton.punct_sentence_simple(masked_sent,bool_tweet_bert)
                
#            print(bool_tweet_bert, 'punct_sentence_tweet: ', time.time()-start_time)

            if len(ent)  == 1:
                new_descs = []
                new_desc_probs = []
                for i, item in enumerate(descs):
                    if str(item).lower() !=  str(ent[0]).lower():
                        new_descs.append(item)
                        new_desc_probs.append(desc_probs[i])
                descs = new_descs
                desc_probs = new_desc_probs
        else:
            if 1 < len(ent):
                # masked_sent = masked_sent.replace(DISPATCH_MASK_TAG,MASK_TAG2)
#                start_time = time.time()
                descs, desc_probs = self.desc_singleton.punct_sentence_simple(masked_sent)
#                print(bool_tweet_bert, 'punct_sentence_simple2: ', time.time()-start_time)
            else:
#                start_time = time.time()
                if ent[0] in self.Word_Entities.keys():
                    return_dic = self.Word_Entities[ent[0]]
                    if 'LOC' not in return_dic.keys():
                        return_dic['LOC'] = 0
#                    print(bool_tweet_bert, 'bert cluster: ', time.time()-start_time)
                    return return_dic, return_dic,descs
                elif ent[0].lower() in self.Word_Entities.keys():
                    return_dic = self.Word_Entities[ent[0].lower()]
                    if 'LOC' not in return_dic.keys():
                        return_dic['LOC'] = 0
#                    print(bool_tweet_bert, 'bert cluster: ', time.time()-start_time)

                    return return_dic, return_dic,descs

                else:
                    
                    masked_sent = ori_masked_sen;#.replace(DISPATCH_MASK_TAG,MASK_TAG2)
#                    start_time = time.time()
                    descs, desc_probs = self.desc_singleton.punct_sentence_simple(masked_sent)
#                    print(bool_tweet_bert, 'punct_sentence_simple2: ', time.time()-start_time)
        if bool_debug:
            print(descs)
      
        sum_pro = 0
        sum_pro_gen = 0
#        sum_entity = []
#        start_time = time.time()
#        pdb.set_trace()
        for j, desc in enumerate(descs):
#            print(desc)
            if bool_tweet_bert and 3 < len(str(desc)):
                desc = desc.capitalize()
            if desc not in self.desc_singleton.descs:
                sum_pro+=desc_probs[j]
#                sum_entity.append("0")
                sum_pro_gen += 1
            else:
                if desc in self.Word_Entities_C.keys() or desc.lower() in self.Word_Entities_C.keys():
                    if desc in self.Word_Entities_C.keys():
                        entities_raw = self.Word_Entities_C[desc]
                    else:
                        entities_raw = self.Word_Entities_C[desc.lower()]
                        
                    if bool_tweet_bert: # 
                        entities = entities_raw
                    else:
                        entities = {}
                        max_ent = max(entities_raw, key=entities_raw.get)
                        if  entities_raw[max_ent] < 0.1:
                            entities['LOC'] = 0
                        else:                            
                            entities[max_ent] = 1
                    for m in entities.keys():
                        if m:
                            if m in ent_prob.keys():
                                ent_prob[m]+= entities[m]*desc_probs[j]
                            else:
                                ent_prob[m] = entities[m]*desc_probs[j]
                                
                            if m in ent_prob_gen.keys():
                                ent_prob_gen[m] += entities[m]
                            else:
                                ent_prob_gen[m] = entities[m]
                                
                        sum_pro+=entities[m]*desc_probs[j]
                        sum_pro_gen += ent_prob_gen[m]
    
                else:
                    sum_pro+=desc_probs[j]
                    sum_pro_gen += 1
    
        if sum_pro:
            for key in ent_prob.keys():
                ent_prob[key]=ent_prob[key]/sum_pro

        if sum_pro_gen:
            for key in ent_prob_gen.keys():
                ent_prob_gen[key]=ent_prob_gen[key]/sum_pro_gen
                
        return ent_prob, ent_prob_gen,descs
    
    def extract_location(self,strings,thres1=0.7,region=-2,\
           special_con_t=0.35, abb_ent_thres=0.3, context_thres=0.3, \
            weight=1,bool_fast=1, special_ent_t=0.5, \
             merge_thres=0.5,\
            fc_ratio=0.25,input_file='test.txt',\
            abb_context_thres=0.2, num_context_thres=0.2, \
            single_person_c_t=0.23,bool_debug=0,bool_formal=0):
        time_str = datetime.now().strftime('%m%d%H%M%S')
        
        F1, P,R, detection_results = place_tagging(0, time_str,self,thres1,100,\
           special_con_t, abb_ent_thres, context_thres, \
            weight,bool_fast, special_ent_t, \
             merge_thres,\
            fc_ratio, input_file,\
            abb_context_thres, num_context_thres, \
            single_person_c_t,bool_debug,bool_formal,strings)
        return detection_results

        
def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--id', type=str, default='0622143005')
    parser.add_argument('--osmembed', type=int, default= 7)
    parser.add_argument('--thres1', type=float, default= 0.7)
    parser.add_argument('--filter_l', type=int, default= 1)
    parser.add_argument('--bool_osm', type=int, default= 0)
    parser.add_argument('--emb', type=int, default= 1)
    parser.add_argument('--cnn', type=int, default= 150)
    parser.add_argument('--lstm', type=int, default= 150)
    parser.add_argument('--special_con_t', type=float, default= 0.35)
    parser.add_argument('--input', type=int, default= 4)
    parser.add_argument('--input_file', type=str, default= 'test.txt')
    parser.add_argument('--epoch', type=int, default= 4)
    parser.add_argument('--abb_ent_thres', type=float, default= 0.3)
    parser.add_argument('--context_thres', type=float, default= 0.3)
    parser.add_argument('--abb_context_thres', type=float, default= 0.2)
    parser.add_argument('--num_context_thres', type=float, default= 0.2)
    parser.add_argument('--single_person_c_t', type=float, default= 0.23)
    parser.add_argument('--osm', type=int, default= 1)
    parser.add_argument('--weight', type=int, default= 1)
    parser.add_argument('--bool_fast', type=int, default= 1)
    parser.add_argument('--special_ent_t', type=float, default= 0.5)
    parser.add_argument('--bool_general_check', type=int, default= 1)
    parser.add_argument('--general_words', type=int, default= 26000)
    parser.add_argument('--merge_thres', type=float, default= 0.5)
    parser.add_argument('--dic_neig', type=int, default= 301)
    parser.add_argument('--con_neig', type=int, default= 301)
    parser.add_argument('--emw', type=int, default= 4)
    parser.add_argument('--fc_ratio', type=float, default= 0.25)
    parser.add_argument('--bool_debug', type=int, default= 0)
    parser.add_argument('--bool_formal', type=int, default= 0)
    parser.add_argument('--c_model', type=int, default= 0)
    parser.add_argument('--i_model', type=int, default= 0)

    args = parser.parse_args()
    if args.bool_debug:
        print ('id: '+str(args.id))
        print ('thres: '+str(args.thres1))
        print ('filter_l: '+str(args.filter_l))
        print ('emb: '+str(args.emb))
        print ('cnn: '+str(args.cnn))
        print ('lstm: '+str(args.lstm))
        print ('special_con_t: '+str(args.special_con_t))
        print ('input: '+str(args.input))
        print ('input_file: '+str(args.input_file))
        print ('epoch: '+str(args.epoch))
        print ('abb_ent_thres: '+str(args.abb_ent_thres))
        print ('context_thres: '+str(args.context_thres))
        print ('abb_context_thres: '+str(args.abb_context_thres))
        print ('num_context_thres: '+str(args.num_context_thres))
        print ('single_person_c_t: '+str(args.single_person_c_t))
    
        print ('osm: '+str(args.osm))
        print ('weight: '+str(args.weight))
        print ('bool_fast: '+str(args.bool_fast))
        print ('special_ent_t: '+str(args.special_ent_t))
        print ('bool_general_check: '+str(args.bool_general_check))
        print ('general_words: '+str(args.general_words))
        print ('merge_thres: '+str(args.merge_thres))
        print ('dic_neig: '+str(args.dic_neig))
        print ('con_neig: '+str(args.con_neig))
        print ('emw: '+str(args.emw))
        print ('fc_ratio: '+str(args.fc_ratio))
        print ('bool_debug: '+str(args.bool_debug))
        print ('bool_formal: '+str(args.bool_formal))
        print ('c_model: '+str(args.c_model))
        print ('i_model: '+str(args.i_model))
    
    start_time = time.time()
    gazpne2 = GazPNE2(args.id,args.epoch, args.lstm, args.filter_l, args.osm, args.osmembed, args.bool_general_check, \
                      args.general_words, args.emb, args.bool_osm, args.dic_neig, args.con_neig, \
                          args.emw, args.c_model, args.i_model)
    # obj = UnsupNER(args.dic_neig,args.con_neig,args.emw,args.c_model,args.i_model)
    # if args.bool_debug:
    #     print('UnsupNER', time.time()-start_time)
    print('model is loading...')
    time_str = datetime.now().strftime('%m%d%H%M%S')
    # print('time_str',time_str)
        
    if args.input == 2:
        regions=[32,31,30]
    elif args.input == 3:
        regions=[-2,-3,-4,-5]

    elif args.input == 0:
        regions=[49]
    elif args.input == -10:
        regions=[5]        
    elif args.input == 5:
        regions=[4]
    elif args.input == 6:
        regions=[50]
    elif args.input == 7:
        regions=[51]
    elif args.input == 8:
        regions=[52]
    elif args.input == 9:
        regions=[53]
    elif args.input == 10:
        regions=[55]
    elif args.input == 11:
        regions=[56]
    elif args.input == 12:
        regions=[57]
    elif args.input == 13:
        regions=[58]
    elif args.input == 14:
        regions=[59]
    else:
        regions = [15,26,28,25,6,7,0,1,2,9,10,20, 21, 8,17,11,12,13,14]
    start_time = time.time()

    # print('load_osm_names_fre', time.time()-start_time)
    # print(general_place_list)
    # Write each dataframe to a different worksheet.
    for file in os.listdir("model/"):
        if args.id in file and '.pkl' in file:
            if 'clstm' in file:
                try:
                    new_name = 'model/'+file[12:len(file)]
                    os.rename('model/'+file, new_name)
                    epoch_str = file[27:len(file)][:-4]
                    epoch = int(epoch_str)
                except BaseException:
                    continue
            else:
                epoch_str = file[15:len(file)][:-4]
                epoch = int(epoch_str)
            # print('epoch:'+str(epoch))
            if epoch == args.epoch:
                for r_idx, region in enumerate(regions):
                    F1, P,R, detection_results = place_tagging(int(args.input == 1), time_str,gazpne2,args.thres1,region,\
                               args.special_con_t, args.abb_ent_thres, args.context_thres, \
                                args.weight,args.bool_fast, args.special_ent_t, \
                                 args.merge_thres,\
                                args.fc_ratio,args.input_file,\
                                args.abb_context_thres, args.num_context_thres, \
                                args.single_person_c_t,args.bool_debug,args.bool_formal)
                    print('region '+ str(region)+' : F1 ' + str(F1)+' : thres ' + str(args.thres1))
#    writer.save()
if __name__ == '__main__':
    main()
