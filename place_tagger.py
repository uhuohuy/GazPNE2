import csv
import json, re
import torch
from rules import * 
from utility import *
import numpy as np
import string
from core import extract_sim,align_and_split,get_removed_indices_mention, align_adv
from gensim.models import KeyedVectors 
from datetime import datetime
from wordsegment import segment
from CMUTweetTagger import runtagger_parse
import time
import json
import pandas as pd
from utility import load_osm_names_fre
import os
import torch.nn.functional as F
import demoji
import stanza

# PATH_TO_JAR='stanford-ner-2015-04-20.jar'
# PATH_TO_MODEL = 'english.all.3class.distsim.crf.ser.gz';'english.muc.7class.distsim.crf.ser.gz';
# tagger = StanfordNERTagger(model_filename=PATH_TO_MODEL,path_to_jar=PATH_TO_JAR, encoding='utf-8')
Indefinite_articles = ['a','an','my','our']

stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

WORD_POS = 0
TAG_POS = 1
noun_tags_tweet = ['N','$','^','A', 'O','G'] #'P',
nan_single = ['A', 'O']
pla_ctx_tags = ['N','$','^','A','G','P','&',',']
not_start = ['P']
not_end = ['P','A']
MASK_TAG = "entity"
MASK_TAG_tweet = '<mask>'
PERSON_POS = ['^','$']
NUMBER_POS = ['$']
IGNORE = [] #'#','NEWLINE'
CAP = ['N','^','G']
LOW = ['V']
spatial_indicators = ['in','near','at','on','to','beyond','over','off','under','behind','from']
exp_pos_list = ['^','$']
exp_word = ['the']
non_LOC_NER_TYPEs = ['PERSON'] #,,'DATE','QUANTITY''PERSON',','ORG'
LOC_NER_TYPEs = ['LOC','FAC','LOCATION','GPE']#'GPE',
non_LOC_NER_TYPEs_2 = ['TIME']
category_words_raw = load_osm_names_fre('data/category_words.txt', [], aug_count = 1)
category_words_raw = [word[0] for word in category_words_raw]
category_words_simple = [item for item in category_words]
category_words.extend(category_words_raw)
category_words = [item for item in category_words if item != 'nyc']
category_words_tuple = [tuple([word]) for word in category_words]
# general_words = list(set(general_words).difference(set(category_words_tuple)))

def lowerize(offsets, full_offset, tag_lists):
    new_off = []
    for s in full_offset:
        bool_lower = 0
        for i, suboff in enumerate(offsets):
            for j, subsuboff in enumerate(suboff):
                if s[1] >= subsuboff[0] and s[1] <= subsuboff[1] and \
                   s[2] >= subsuboff[0] and s[2] <= subsuboff[1]:
                       if tag_lists[i][j][1] in LOW:
                           bool_lower = 1
                       break
            if bool_lower:
                break
        if bool_lower:
            new_off.append(tuple([s[0].lower(), s[1], s[2]])) #.capitalize()
        else:
            new_off.append(tuple([s[0], s[1], s[2]]))
    return new_off


                 
def extract_nouns_tweet(terms_arr,max_len, dis_split,cur_off, bool_keep_con = 0):
    new_arr = []
    for item in terms_arr:
#        print(item)
        new_item = list(item)
        if item[2] < 0.4 or item[0].lower()=='of':
            new_item[1] = noun_tags_tweet[0]
        new_arr.append(new_item)
    span_arr, noun_array, indexs = generate_nouns(new_arr)
    return_list = []
    return_list_index = []
    return_pos = []
    for i, sublist in enumerate(noun_array):

        cur_list, cur_index = sub_lists_pos_adv(sublist,indexs[i], new_arr, max_len)
        bool_valid = True
        if bool_keep_con:
            cur_off_pla = tuple([cur_off[cur_index[0]][0],cur_off[cur_index[-1]][1]])
            for split in dis_split:
                if overlap(list(range(split[0],split[1]+1)), list(range(cur_off_pla[0],cur_off_pla[1]+1))):
                    bool_valid = False
                    break
        if bool_valid:
            return_list.extend(cur_list)
            return_list_index.extend(cur_index)
            for c_index in cur_index:
                return_pos.append([new_arr[index][1] for index in c_index])
    
    return return_list_index, return_list,return_pos

def gen_mask_sentence3(sentence):
    sen = ''
    cap_list =  []
    origin_sen = ''
    if len(sentence)==1 and len(sentence[0]) <= 3:
        bool_cap = True
    else:
        bool_cap = False
    for i, s in enumerate(sentence):
        if not bool_cap:
            sen += s.capitalize() + ' '
            cap_list.append(s.capitalize())
        else:
            sen += s + ' '
            cap_list.append(s)
        origin_sen += s + ' '
    masked_sen = sen + 'is a ' + MASK_TAG +'.'
    ori_masked_sen = origin_sen + 'is a ' + MASK_TAG+'.'
    return masked_sen, cap_list, ori_masked_sen



def gen_mask_sentence2(detected_offset, ori_offset,  pos_list,  max_words=80):
    sen = ''
    bool_inser = 0
    mask_index = 0
#    if len(ori_offset) > max_words:
    for i, s in enumerate(ori_offset):
        if s[1] >= detected_offset[0] and s[1] <= detected_offset[1] and \
            s[2] >= detected_offset[0] and s[2] <= detected_offset[1] :
            mask_index = i
            break
    limited_x = 0
    limited_y = len(ori_offset)
    if mask_index-limited_x > int(max_words/2):
        limited_x = mask_index-int(max_words/2)
    if limited_y - mask_index > int(max_words/2):
        limited_y = mask_index+int(max_words/2)
    for i in range(limited_x, limited_y):
        s = ori_offset[i]
#    for i, s in enumerate(ori_offset):
        if s[1] >= detected_offset[0] and s[1] <= detected_offset[1] and \
            s[2] >= detected_offset[0] and s[2] <= detected_offset[1] :
            if not bool_inser:
                mask_index = i
                sen += ' ' + MASK_TAG_tweet + ' '   
            bool_inser = 1
        else:
            if s[0] not in IGNORE:
                if s[0] == '#' and s[2]== detected_offset[0]-1:
                    continue
                sen += s[0]
                if i< limited_y-1: 
                    if s[2]+1 != ori_offset[i+1][1]:
                        sen +=  ' '
    return sen



def capitalize(terms_arr):
    for i,term_tag in enumerate(terms_arr):
        #print(term_tag)
        if (term_tag[TAG_POS] in cap_tags):
            word = term_tag[WORD_POS][0].upper() + term_tag[WORD_POS][1:]
            term_tag[WORD_POS] = word
    #print(terms_arr)


def generate_nouns(terms_arr):
    size = len(terms_arr)
    span_arr = []
    indexs = []
    i = 0
    return_results = []
    while (i < size):
        term_info = terms_arr[i]
        if (term_info[TAG_POS] in noun_tags_tweet and term_info[TAG_POS] not in not_start):
            skip = gen_sentence_raw(terms_arr,i)
            if skip==1 and term_info[TAG_POS] in nan_single:
                i += 1
                span_arr.append(0)
            else:
                i +=  skip
                temp = []
                for j in range(skip):
                    span_arr.append(1)
                    temp.append(terms_arr[i-skip+j][WORD_POS])
    #                temp_pos.append(terms_arr[i-skip+j][TAG_POS])
                return_results.append(temp)
    #            pos_result.append(temp_pos)
                indexs.append(list(range(i-skip,i)))
        else:
            i += 1
            span_arr.append(0)
    #print(sentence_arr)
    return span_arr, return_results, indexs#, pos_result


def sub_lists_pos_adv(list1, list_index, tags, max_len): 
    # store all the sublists  
    sublist = []
    sublist_index = []   
    # first loop  
    for i in range(len(list1) + 1):         
        # second loop  
        for j in range(i + 1, len(list1) + 1):             
            # slice the subarray
            if j-i<max_len:
                if tags[list_index[i]][TAG_POS] not in not_start \
                and  tags[list_index[j-1]][TAG_POS] not in not_end and \
                not (j-i==1 and tags[list_index[i]][TAG_POS] in nan_single):
                    sub = list1[i:j]
                    sub_index = list_index[i:j]
                    sublist_index.append(sub_index)
                    sublist.append(sub)
    return sublist,sublist_index

def gen_sentence_raw(terms_arr,index):
    size = len(terms_arr)
    i = index
    skip = 0
    while (i < size):
        if (terms_arr[i][TAG_POS] in noun_tags_tweet):
            skip += 1
            i += 1
        else:
            break
    j = i-1
    while j > index:
        if (terms_arr[j][TAG_POS] in not_end):
            skip -= 1
            j -= 1
        else:
            break
    return skip


'''load the bigram model from file'''

# ''' get the embedding of a sentence  '''
# def sentence_embeding(sentence, trained_emb, word_idx,glove_emb,osm_emb,\
#                       max_len,emb_dim,gaz_emb_dim,\
#                       max_char_len,bool_mb_gaze,\
#                       PAD_idx,START_WORD,listOfProb, char_hc_emb,flex_feat_len):
#     matrix_len = len(sentence)
#     weights_matrix = np.zeros((max_len, emb_dim+gaz_emb_dim+6+flex_feat_len)); 

#     for i in range(0,max_len-matrix_len):
#         char_loc_feat = []
#         if flex_feat_len - 3 > 0:
#             char_loc_feat = [0]*(flex_feat_len - 3)
#         weights_matrix[i] = np.concatenate((trained_emb[PAD_idx],[0,0,0],char_loc_feat),axis=None)
#     for i, word in enumerate(sentence):
#         temp_hc = []
#         temp_hc.append(len(sentence))
#         temp_hc.append(i+1)
#         if i==0:
#             pre_word = START_WORD
#         else:
#             pre_word = sentence[i-1]
#         try:
#             temp_hc.append(listOfProb[(word, pre_word)])
#         except KeyError:
#             temp_hc.append(0)
#         if flex_feat_len - 3 > 0:
#             char_loc_feat = feat_char_loc(word, max_char_len)
#             temp_hc.extend(char_loc_feat)
#         if word not in word_idx.keys():
#             try: 
#                 temp_glove = glove_emb[word]
#             except KeyError:
#                 temp_glove = np.random.normal(scale=0.1, size=(emb_dim,))
#             if bool_mb_gaze:
#                 try: 
#                     temp_gaz = osm_emb[word]
#                 except KeyError:
#                     temp_gaz = np.random.normal(scale=0.1, size=(gaz_emb_dim,))
#             else:
#                 temp_gaz = []
#             try:
#                 temp_hc6 = char_hc_emb[word]
#             except KeyError:
#                 temp_hc6 = np.random.normal(scale=2, size=6)
#             weights_matrix[i+max_len-matrix_len]=np.concatenate((temp_glove,temp_gaz,temp_hc6,np.asarray(temp_hc)
# ), axis=None)
#         else:
#             w_idx = word_idx[word]
#             weights_matrix[i+max_len-matrix_len]=np.concatenate((trained_emb[w_idx],np.asarray(temp_hc)),axis=None)

#     return weights_matrix




''' get the embedding of a sentence  '''
def sentence_embeding(sentence, trained_emb, word_idx,glove_emb,osm_emb,\
                      max_len,emb_dim,gaz_emb_dim,\
                      max_char_len,bool_mb_gaze,\
                      PAD_idx,START_WORD,listOfProb, char_hc_emb,flex_feat_len,bool_hc=1,bool_fix_hc=1):
    matrix_len = len(sentence)
    weights_matrix = np.zeros((max_len, emb_dim+gaz_emb_dim+6*bool_hc+flex_feat_len)); 

    for i in range(0,max_len-matrix_len):
        char_loc_feat = []
        if flex_feat_len - 3 > 0:
            char_loc_feat = [0]*(flex_feat_len - 3)
        if bool_hc:
            default_hc = [0,0,0]
        else:
            default_hc = []
            
        weights_matrix[i] = np.concatenate((trained_emb[PAD_idx],default_hc,char_loc_feat),axis=None)
    for i, word in enumerate(sentence):
        temp_hc = []
        if bool_hc:
            temp_hc.append(len(sentence))
            temp_hc.append(i+1)
        if i==0:
            pre_word = START_WORD
        else:
            pre_word = sentence[i-1]
        if bool_hc:
            try:
                temp_hc.append(0)
                #temp_hc.append(listOfProb[(word, pre_word)])
            except KeyError:
                temp_hc.append(0)
            if flex_feat_len - 3 > 0:
                char_loc_feat = feat_char_loc(word, max_char_len)
                temp_hc.extend(char_loc_feat)
        if word not in word_idx.keys():
            try: 
                temp_glove = glove_emb[word]
            except KeyError:
                temp_glove = np.random.normal(scale=0.1, size=(emb_dim,))
            if bool_mb_gaze:
                try: 
                    temp_gaz = osm_emb[word]
                except KeyError:
                    temp_gaz = np.random.normal(scale=0.1, size=(gaz_emb_dim,))
            else:
                temp_gaz = []
            temp_hc6 = []
            if bool_hc and bool_fix_hc:
                try:
                    temp_hc6 = char_hc_emb[word]
                except KeyError:
                    temp_hc6 = np.random.normal(scale=2, size=6)
            weights_matrix[i+max_len-matrix_len]=np.concatenate((temp_glove,temp_gaz,temp_hc6,np.asarray(temp_hc)
), axis=None)
        else:
            w_idx = word_idx[word]
            weights_matrix[i+max_len-matrix_len]=np.concatenate((trained_emb[w_idx],np.asarray(temp_hc)),axis=None)

    return weights_matrix

 

def extract_subtaglist(tag_list, full_offset, sub_offset):
    return_list = []
    last_match = 0
    for s in sub_offset:
        for i, subsuboff in enumerate(full_offset):
            if i >= last_match:
                if subsuboff[1] >= s[0]  and subsuboff[1]  <= s[1] and \
                subsuboff[2] >= s[0]  and subsuboff[2]  <= s[1] :
                    return_list.append(tag_list[i])
                    last_match = i+1
                    break
    return return_list

'''deal with the 's issue.'''
def align(tags, full_offset):
    new_offsets = []
    last_index = 0
    for i, offset in enumerate(full_offset):
        if offset[0]==tags[last_index][0]:
            new_offsets.append(offset)
            last_index += 1
        else:
            last_left = len(offset[0])
            while last_left > 0:
                start_p = offset[1] + len(offset[0])-last_left
                end_p = start_p+len(tags[last_index][0])-1
                new_offsets.append(tuple([tags[last_index][0],start_p,end_p]))
                last_left = last_left-len(tags[last_index][0])
                last_index += 1
    return new_offsets

def load_cache_from_file(region,word2idx,abv_punk,input_file,strings=[]):
#    bert_cache = {}
#    ignored_place_named = {}
#    ignored_place_named[6] = [('new','york'),('nyc',),('new','york','city'),('ny')]
#    ignored_place_named[7] = [('new','zealand'),('nz',),('uk'),('christchurch',),('chch',),('lyttleton',), ('southland',),('wellington',), ('south','island')]
#    if region in ignored_place_named.keys(): 
#        ignored_places = ignored_place_named[region]
#    else:
#        ignored_places = []
    raw_data_dir = 'data/events_set'
#    raw_data_dir = ''
    target_dir = []
    if region==-1:
        t_json_file = "data/test_data/formal/GWN_preproceed.json"#"data/raw_tweet.txt"
    elif region==-2:
        t_json_file = "data/test_data/formal/lgl_preproceed.json"#"data/raw_tweet.txt"
    elif region==-3:
        t_json_file = "data/test_data/formal/TR-News_preproceed.json"#"data/raw_tweet.txt"
    elif region==-4:
        t_json_file = "data/test_data/formal/GeoVirus_preproceed.json"#"data/raw_tweet.txt"
    elif region==-5:
        t_json_file = "data/test_data/formal/.json"#"data/raw_tweet.txt"
    elif region==1:
        t_json_file = "data/test_data/houston_floods_2016_annotations.json"#"data/raw_tweet.txt"
    elif region==2:
        t_json_file = "data/test_data/chennai_floods_2015_annotations.json"#"
    elif region==0:
        t_json_file = "data/test_data/louisiana_floods_2016_annotations.json"#"
    elif region==6:
        t_json_file = "data/test_data/benchmark_ny_annotated.json"#"
    elif region==7:
        t_json_file = "data/test_data/benchmark_nz_annotated.json"#"
    elif region==8:
        t_json_file = "data/test_data/geocorpora_total.json"#"
    elif region==9:
        t_json_file = "data/test_data/b.json"#"
    elif region==10:
        t_json_file = "data/test_data/e.json"#"
    elif region==11:
        t_json_file = "data/test_data/f.json"#"
    elif region==12:
        t_json_file = "data/test_data/g.json"#"
    elif region==13:
        t_json_file = "data/test_data/h.json"#"
    elif region==14:
        t_json_file = "data/test_data/a.json"#"
    elif region==15:
        t_json_file = "data/test_data/HarveyTweet2017.json"#"
    elif region==17:
        t_json_file = "data/test_data/microposts2016-neel-training_neel.json"#"
    elif region==20:
        t_json_file = "data/test_data/train.json"#"
    elif region==21:
        t_json_file = "data/test_data/ritter_ner.json"#"
    elif region==25:
        t_json_file = "data/test_data/tweet_dataset_I.json"#"
    elif region==26:
        t_json_file = "data/test_data/tweet_dataset_II.json"#"
    elif region==28:
        t_json_file = "data/test_data/msm2013train.json"#"
    elif region==30:
        t_json_file = "data/test_data/humaid.json"#"
    elif region==31:
        t_json_file = "data/test_data/crisisbench.json"#"
    elif region==32:
        t_json_file = "data/test_data/corona.json"#"

#    elif region == 4:
#        dir_list = [os.path.join(raw_data_dir, o) for o in os.listdir(raw_data_dir) 
#                        if os.path.isdir(os.path.join(raw_data_dir,o))]
    elif region== 5:
         dir_list = ['data']
    elif region ==-10:
        raw_data_dir = 'data/D4h_data'
        target_dir = [raw_data_dir]
    elif region == 50:
        raw_data_dir = 'data/events_set'
    elif region == 51:
        raw_data_dir = 'data/all_data_en'
        target_dir = [raw_data_dir]
    elif region == 52:
        raw_data_dir = 'data/all_data_en2'
        target_dir = [raw_data_dir]
    elif region == 53:
        raw_data_dir = 'data/all_data_en3'
        target_dir = [raw_data_dir]
    elif region == 55:
        raw_data_dir = 'data/coronavirus/cv1'
    elif region == 56:
        raw_data_dir = 'data/coronavirus/cv2'
    elif region == 57:
        raw_data_dir = 'data/coronavirus/cv3'
    elif region == 58:
        raw_data_dir = 'data/coronavirus/cv4'
    elif region == 59:
        raw_data_dir = 'data/coronavirus/cv5'
    if not target_dir:
        target_dir = [os.path.join(raw_data_dir, o) for o in os.listdir(raw_data_dir) 
                        if os.path.isdir(os.path.join(raw_data_dir,o))]
    test_keys = []
    tweet_cache = {}
    org_ignore_list = [18,19,20,21]
    total_tweet_count = 0
    '''preload data to cache'''
    if region in [4,5,-10]:
        file_names = []
        if region==-10:
            for sub_dir in target_dir:
                temp_file_names = [f for f in os.listdir(sub_dir) if f.endswith('.ndjson')]
                file_names.extend(temp_file_names)
        else:
            for sub_dir in dir_list:
                if region == 4:
                    new_file_name = sub_dir+'/tweets.jsonl'
                else:
                    new_file_name = sub_dir+'/matti_tweets.ndjson'
                file_names.append(new_file_name)
        for new_file_name in file_names:
            if os.path.isfile(new_file_name):
                jsonObj = pd.read_json(path_or_buf=new_file_name, lines=True)
                for i in range(len(jsonObj)):
                    if region == -10:
                        tweet = jsonObj['translation'][i].encode("ascii", "ignore").decode("utf-8")
                    else:
                        tweet = jsonObj['text'][i].encode("ascii", "ignore").decode("utf-8")
                    key = jsonObj['id'][i]
                    place_names = []
                    place_offset = []
                    amb_place_names = []
                    amb_place_offset = []

                    test_keys.append(key)
                    if key not in tweet_cache.keys():
                        sentences, offsets,full_offset,hashtag_offsets,dis_split = extract_sim(tweet,word2idx.keys(),1,abv_punk)
                        hashtag_offsets = []
                        sentences_lowcases = [[x.lower() for x in y] for y in sentences]
                        tweet_cache[key]=[place_names,place_offset,sentences,offsets,full_offset,sentences_lowcases,tweet,hashtag_offsets,dis_split,amb_place_names,amb_place_offset]
    elif region in [55,56,57,58,59]:
        # print(target_dir)
        for sub_dir in target_dir:
            json_files = [f for f in os.listdir(sub_dir) if f.endswith('.json')]
            # print(json_files)
            for file in json_files:
                json_file = sub_dir+'/'+file
                jf = open(json_file)
                js_data = json.load(jf)
                for key in js_data.keys():
                    tweet = js_data[key]['text']
                    test_keys.append(key)
                    if key not in tweet_cache.keys():
                        sentences, offsets,full_offset,hashtag_offsets,dis_split = extract_sim(tweet,word2idx.keys(),1,abv_punk)
                        hashtag_offsets = []
                        sentences_lowcases = [[x.lower() for x in y] for y in sentences]
                        tweet_cache[key]=[[],[],sentences,offsets,full_offset,sentences_lowcases,tweet,hashtag_offsets,dis_split,[],[]]

    elif region in [50, 51,52,53]:
        for sub_dir in target_dir:
            tsv_files = [f for f in os.listdir(sub_dir) if f.endswith('.tsv')]
            for file in tsv_files:
                new_file_name = sub_dir+'/'+file
#                    print(new_file_name)
                if os.path.isfile(new_file_name):
                    df = pd.read_csv(new_file_name,sep='\t',quoting=csv.QUOTE_NONE, encoding='utf-8')
                    for i in range(len(df)):
                        if region == 50:
                            tweet = df['tweet_text'][i].encode("ascii", "ignore").decode("utf-8")
                            key = df['tweet_id'][i]
                        else:
                            tweet = df['text'][i].encode("ascii", "ignore").decode("utf-8")
                            key = df['id'][i]
                        test_keys.append(key)
                        if key not in tweet_cache.keys():
                            sentences, offsets,full_offset,hashtag_offsets,dis_split = extract_sim(tweet,word2idx.keys(),1,abv_punk)
                            hashtag_offsets = []
                            sentences_lowcases = [[x.lower() for x in y] for y in sentences]
                            tweet_cache[key]=[[],[],sentences,offsets,full_offset,sentences_lowcases,tweet,hashtag_offsets,dis_split,[],[]]
    elif region == 49 or region == 100:
        if region == 100:
            Lines = strings
        else:
            file1 = open(input_file, 'r')
            Lines = file1.readlines()
        for i, tweet in  enumerate(Lines):
            tweet = tweet.strip()
            if not tweet:
                continue
            key = i
            test_keys.append(key)
            if key not in tweet_cache.keys():
                sentences, offsets,full_offset,hashtag_offsets,dis_split = extract_sim(tweet,word2idx.keys(),1,abv_punk)
                hashtag_offsets = []
                sentences_lowcases = [[x.lower() for x in y] for y in sentences]
                tweet_cache[key]=[[], [],sentences,offsets,full_offset,sentences_lowcases,tweet,hashtag_offsets,dis_split, [], []]
    else:
        if  os.path.isfile(t_json_file):
            with open(t_json_file) as json_file:
                js_data = json.load(json_file)
                for key in js_data.keys():
                    tweet = js_data[key]['text']
                    tweet = demoji.replace(tweet,  " ") 
                    place_names = []
                    place_offset = []
                    amb_place_names = []
                    amb_place_offset = []
    
                    total_tweet_count += 1
                    test_keys.append(key)
                    for cur_k in js_data[key].keys():
                        if cur_k == 'text':
                            tweet = js_data[key][cur_k]
                        else:
                            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", js_data[key][cur_k]['text'])         
                            corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
                            corpus = [word  for word in corpus if word]
                            if js_data[key][cur_k]['type'] != 'ambLoc':
                                place_names.append(tuple(corpus))
                                place_offset.append(tuple([int(js_data[key][cur_k]['start_idx']),int(js_data[key][cur_k]['end_idx'])-1]))
                            else:
                                if region in org_ignore_list:
                                    amb_place_names.append(tuple(corpus))
                                    amb_place_offset.append(tuple([int(js_data[key][cur_k]['start_idx']),int(js_data[key][cur_k]['end_idx'])-1]))
    
                    if key not in tweet_cache.keys():
                        sentences, offsets,full_offset, hashtag_offsets ,dis_split = extract_sim(tweet,[],1,abv_punk)
                        sentences_lowcases = [[x.lower() for x in y] for y in sentences]
                        tweet_cache[key]=[place_names,place_offset,sentences,offsets,full_offset,sentences_lowcases,tweet,hashtag_offsets,dis_split,amb_place_names,amb_place_offset]
    # print('tag pos')
    index = 0
    tag_list = []
    total_sen = []
#    start_time = time.time()
    for key in test_keys:
        sen = ''
        for sent in tweet_cache[key][4]:
            sen += sent[0] + ' '
        total_sen.append(sen)
        index += 1
        if index == 1000:
            tag_list_temp = runtagger_parse(total_sen)
            tag_list.extend(tag_list_temp)
            index=0
            total_sen = []
    if index > 0:
        tag_list_temp = runtagger_parse(total_sen)
        tag_list.extend(tag_list_temp)
    tag_list=[item for item in tag_list if item]
    # print('tag pos done')
    # print(len(tag_list))
#            print('pos time',time.time()-start_time)
    
#    start_time = time.time()

    temp_tags = {}
    valid_keys = []
    index = 0
    for key in test_keys:
        tag_lists = []
        # print(key)
        # print(tweet_cache[key][4])
        # print(index)
        
        if tweet_cache[key][4]: # and tag_list[index]
            # print(tag_list[index])
            aligned_full_offset = align(tag_list[index], tweet_cache[key][4])
#                    print(aligned_full_offset)

            for i in range(len(tweet_cache[key][2])):
                temp_offset = extract_subtaglist(tag_list[index], aligned_full_offset, tweet_cache[key][3][i])#
                tag_lists.append(temp_offset)
            temp_tags[key]= tag_lists
            valid_keys.append(key)
            index += 1
    return tweet_cache, valid_keys, temp_tags,total_tweet_count

def create_result(key, p_type, place,pos, cur_off_pla,ent_prob,context_ent_prob,ent_prob_gen,context_ent_prob_gen,\
                  bool_general,bool_person, fc_thres, pos_prob,amb_place_offset, amb_place_names, cur_hash):
    result_mid = {}
    result_mid['ID'] = str(key)
    name = ''
    for t in place:
        name += t + ' '
    pos_name = ''
    for t in pos:
        pos_name += t + ' '

    result_mid['text'] = name
    result_mid['type'] = p_type
    result_mid['general'] = int(bool_general)
    result_mid['pos_prob'] = pos_prob
    result_mid['amb_place_offset'] = str(amb_place_offset)
    result_mid['amb_place_names'] = str(amb_place_names)
    result_mid['cur_hash'] = str(cur_hash)

    result_mid['pos'] = pos_name
    result_mid['fc_thres'] = fc_thres

    result_mid['s_idx'] = cur_off_pla[0]
    result_mid['e_idx'] = cur_off_pla[1]                           
    for ent_k in ent_prob.keys():
        result_mid['in'+ent_k] = ent_prob[ent_k] 
    for ent_k in context_ent_prob.keys():
        result_mid['ex'+ent_k] = context_ent_prob[ent_k] 
    for ent_k in ent_prob_gen.keys():
        result_mid['ingen'+ent_k] = ent_prob_gen[ent_k] 
    for ent_k in context_ent_prob_gen.keys():
        result_mid['exgen'+ent_k] = context_ent_prob_gen[ent_k]
    result_mid['bool_person'] = bool_person
    return result_mid

def top_ent(target_ent, top_n):
    output_prob = {}
    max_ent = len(target_ent.keys())
    if top_n > max_ent:
        top_n = max_ent
    sorted_ent_keys = sorted(target_ent.keys(), key=lambda k: target_ent[k], reverse=True)
    for t in range(top_n):
        output_prob[sorted_ent_keys[t]] = target_ent[sorted_ent_keys[t]]
    if 'LOC' not in sorted_ent_keys:
        output_prob['LOC'] = target_ent['LOC']
    return output_prob

def pure_ent(target_ent):
    output_prob = {}
    for key in target_ent.keys():
        if not key.startswith('OTHER'):
            output_prob[key]=target_ent[key]
    return output_prob

#def include_suffix(cur_off,sub_index,i,j,sub_sen,all_sub_lists,lastfix_places_words):
#    if cur_off[sub_index[j][0]][0] == cur_off[sub_index[i][0]][0] \
#    and cur_off[sub_index[j][-1]][1] > cur_off[sub_index[i][-1]][1] \
#    and len(sub_sen) -1 == len(all_sub_lists[i]) and sub_sen[-1] in lastfix_places_words:
#        return True
#    else:
#        return False
def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) <= min(x2,y2)


def filter_invalid_candidates(sub_index, all_sub_lists, pos_lists,cur_off, stanza_ents):
    invalids = []
    for i, index in enumerate(sub_index):
        start_off = cur_off[index[0]][0]
        end_off = cur_off[index[-1]][1]
        for ent in stanza_ents:
            if ent.type in non_LOC_NER_TYPEs and \
                is_overlapping(ent.start_char,ent.end_char-1,start_off,end_off):
                invalids.append(i)
                break
    sub_index = [sub_index[i] for i in range(len(sub_index)) if i not in invalids]
    all_sub_lists = [all_sub_lists[i] for i in range(len(all_sub_lists)) if i not in invalids]
    pos_lists = [pos_lists[i] for i in range(len(pos_lists)) if i not in invalids]
    return sub_index, all_sub_lists,pos_lists


def filter_invalid_candidates_adv(sub_index, all_sub_lists, pos_lists,cur_off, stanza_ents):
    invalids = []
    for i, index in enumerate(sub_index):
        start_off = cur_off[index[0]][0]
        end_off = cur_off[index[-1]][1]
        for ent in stanza_ents:
            if ent.type in non_LOC_NER_TYPEs:
                terms = ent.text.lower().split()
                if ent.start_char<=start_off and ent.end_char-1>=end_off or \
                    (is_overlapping(ent.start_char,ent.end_char-1,start_off,end_off) and \
                     all_sub_lists[i] in terms):
                     # all_sub_lists[i][0].lower()==terms[0].lower()):
                    if len(terms) <= 2: # and len(all_sub_lists[i])==1
                        invalids.append(i)
                        # print('stanza_person:',all_sub_lists[i])
                        break
            elif ent.type in non_LOC_NER_TYPEs_2:
                if is_overlapping(ent.start_char,ent.end_char-1,start_off,end_off):
                     # all_sub_lists[i][0].lower()==terms[0].lower()):
                    # if len(terms) <= 2: # and len(all_sub_lists[i])==1
                    invalids.append(i)
                    # print('stanza_invalids:', all_sub_lists[i])
                    break

    sub_index = [sub_index[i] for i in range(len(sub_index)) if i not in invalids]
    all_sub_lists = [all_sub_lists[i] for i in range(len(all_sub_lists)) if i not in invalids]
    pos_lists = [pos_lists[i] for i in range(len(pos_lists)) if i not in invalids]
    return sub_index, all_sub_lists,pos_lists


def bool_expansion(sub_index,i,j,lastfix_places_words,final_sub_sen,tag_lists,spatial_indicators,prefix_place_words,exp_pos_list):
    if (is_Sublist(sub_index[j],sub_index[i])):
        if (sub_index[j][0] != 0 and tag_lists[sub_index[j][0]-1][0].lower() in spatial_indicators) or \
           (sub_index[j][0] > 1 and tag_lists[sub_index[j][0]-1][0].lower() in exp_word and \
           tag_lists[sub_index[j][0]-2][0].lower() in spatial_indicators) :
            for k in sub_index[j]:
               if k < sub_index[i][0] and tag_lists[k][1] not in exp_pos_list and tag_lists[k][0].lower() not in prefix_place_words:
                   return False
               elif k> sub_index[i][-1] and tag_lists[k][0].lower() not in lastfix_places_words:
                   return False
        else:
            for k in sub_index[j]:
               if k < sub_index[i][0] and tag_lists[k][0].lower() not in prefix_place_words:
                   return False
               elif k> sub_index[i][-1] and tag_lists[k][0].lower() not in lastfix_places_words:
                   return False
        return True
    else:
        return False


def valid_stanza_places(stanza_ents, detected_places, detected_offset, keys,NER_TYPEs=LOC_NER_TYPEs, word_len=10):
    new_offsets = []
    new_places = []
    for i, ent in enumerate(stanza_ents):
        terms = ent.text.lower().split()
        if len(terms) > word_len:
            continue
        bool_continue = True
        for t in terms:
            if ent.type in NER_TYPEs:
                if t not in keys and t.lower() not in keys and not hasNumbers(t) and  t not in string.punctuation:
                    bool_continue = False
                    break
        if bool_continue and  ent.type in NER_TYPEs:
            if ent.text.lower().startswith('the '):
                start_char = ent.start_char + 4
                place_text = ent.text.lower()[4:len(ent.text.lower())]
            else:
                start_char = ent.start_char
                place_text = ent.text.lower()
            end_char = ent.end_char-1
            place = place_text.split()
            place = [item for item in place if item]
            bool_overlapp = False
            for off in detected_offset:
                if is_overlapping(off[0], off[1], start_char, end_char):
                    bool_overlapp = True
                    break
            if not bool_overlapp:
                new_places.append(place)
                new_offsets.append((start_char,end_char))
    return new_offsets, new_places
       
def inStanzaLoc(stanza_ents, place_name, offset, special=0, NER_TYPEs=LOC_NER_TYPEs, Word_len = 10):
    bool_matched = False
    for i, ent in enumerate(stanza_ents):
        if ent.type in LOC_NER_TYPEs:
            if ent.text.lower().startswith('the '):
                start_char = ent.start_char + 4
                place_text = ent.text.lower()[4:len(ent.text.lower())]
            else:
                start_char = ent.start_char
                place_text = ent.text.lower()
            end_char = ent.end_char-1    
            place = place_text.split()
            place = [item for item in place if item and item not in string.punctuation]
            if len(place)>Word_len:
                continue
            if special:
                if (start_char==offset[0] and end_char == offset[1]) or \
                    (is_overlapping(offset[0], offset[1], start_char, end_char) and place == place_name):
                        bool_matched = True
                        break
            else:
                if (start_char<=offset[0] and end_char >= offset[1]) or \
                    (is_overlapping(offset[0], offset[1], start_char, end_char) and place == place_name):
                        bool_matched = True
                        break
    return bool_matched
                        
    
def filterArticles(sentence,sub_index, all_sub_lists, pos_lists):
    invalids = []
    for i in range(len(sub_index)):
        if sub_index[i][0] > 0 and len(all_sub_lists[i]) > 1 and sentence[sub_index[i][0]-1].lower() in Indefinite_articles:
            invalids.append(i)
    sub_index = [sub_index[i] for i in range(len(sub_index)) if i not in invalids]
    all_sub_lists = [all_sub_lists[i] for i in range(len(all_sub_lists)) if i not in invalids]
    pos_lists = [pos_lists[i] for i in range(len(pos_lists)) if i not in invalids]
    return sub_index, all_sub_lists, pos_lists
    
def sub_stanza(positives,stanza_ents,cur_off,sub_index,all_sub_lists):
    removed = []
    valids = []
    for i in positives:
        offset = tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]])
        place_name = [item.lower() for item in all_sub_lists[i]]
        for j, ent in enumerate(stanza_ents):
            if ent.type in LOC_NER_TYPEs:
                start_char = ent.start_char
                place_text = ent.text.lower()
                for k in range(j+1,len(stanza_ents)):
                    if stanza_ents[k].type in LOC_NER_TYPEs:
                        end_char = stanza_ents[k].end_char-1
                        place_text += ' '+stanza_ents[k].text.lower()
                        place = place_text.split()
                        place = [item for item in place if item and item not in string.punctuation]
                        if (start_char==offset[0] and end_char == offset[1]) or \
                            (is_overlapping(offset[0], offset[1], start_char, end_char) and place == place_name):
                                removed.append(i)
                                valids.extend(list(range(j,k+1)))
                                break
    return removed, list(set(valids))

def general_check(bert_cache,obj,cur_off_pla,special_ent_t,\
                   loc_thres,all_sub_list,new_full_offset,\
                       pos_list,bool_debug,weight,nur_context=0):
    masked_sentence,cap_sen, ori_masked_sen = gen_mask_sentence3(all_sub_list)    
    if not nur_context:
        if masked_sentence in bert_cache.keys():
            ent_prob = bert_cache[masked_sentence][0]
            ent_prob_gen = bert_cache[masked_sentence][1]
            ent_prob = pure_ent(ent_prob)
            if ent_prob['LOC'] > special_ent_t:
                return 1,bert_cache
            
    # if len(cap_sen) < 2 and (cap_sen[0] in obj.Word_Entities.keys() or cap_sen[0].lower() in obj.Word_Entities.keys()):
    #     ent_prob, ent_prob_gen, descs = obj.context_cue_new(masked_sentence,0,cap_sen,ori_masked_sen,bool_debug,0)
    #     ent_prob = pure_ent(ent_prob)
    #     if ent_prob['LOC'] < special_ent_t:
    #         break
    masked_context_sentence = gen_mask_sentence2(cur_off_pla, new_full_offset, [])
    context_ent_prob, context_ent_prob_gen,context_descs = obj.context_cue_new(\
                              masked_context_sentence,1,cap_sen,'',bool_debug,0)                                        
    context_ent_prob = pure_ent(context_ent_prob)
    if bool_debug:                                            
        print('special check:context_ent_prob', context_ent_prob)
    
    if not weight:
        context_ent_prob = context_ent_prob_gen
    
    if (context_ent_prob['LOC'] >= loc_thres):
            return 1, bert_cache        
    else:
        if nur_context:
            return 0, bert_cache
        masked_sentence,cap_sen, ori_masked_sen = gen_mask_sentence3(all_sub_list)
        if masked_sentence in bert_cache.keys():
            ent_prob = bert_cache[masked_sentence][0]
            ent_prob_gen = bert_cache[masked_sentence][1]
        else:
            ent_prob, ent_prob_gen, descs = obj.context_cue_new(masked_sentence,0,\
                                             cap_sen,ori_masked_sen,bool_debug,0)
            bert_cache[masked_sentence] =  [ent_prob,ent_prob_gen]
        ent_prob = pure_ent(ent_prob)
        if bool_debug:                                                                                            
            print('special check:ent_prob', ent_prob)
    
        if ent_prob['LOC'] >= special_ent_t:
            return 1, bert_cache        
        else:
            return 0, bert_cache        


def place_tagging(no_bert, time_str,gazpne2, thres, region,\
              loc_thres=0.1,\
             ent_thres=0.3, context_thres=0.2,  weight=1, \
             bool_fast = 0, special_ent_t = 0.4, \
                 merge_thres=0.4, \
            fc_ratio=0.4, input_file='data/test.txt',abb_context_thres=0.2, \
                num_context_thres=0.2, single_person_c_t=0.2, bool_debug=1, bool_formal=0, strings=[]):
    fc_tokens = gazpne2.fc_tokens
    fc_tokens = [item for item in fc_tokens if item not in category_words_simple]
    bool_special_check = 1
    postive_pro_t = thres
    result_mids = []
    truth_all = []
    abv_punk = gazpne2.abv_punk
    general_words = gazpne2.general_words
    # general_words.extend(['0','00','000','0000'])
    # general_words = list(set(general_words).difference(set([tuple(['s'])])))
    abbr_dict = gazpne2.abbr_dict
    word2idx = gazpne2.word2idx
#     PAD_idx = 0
#     s_max_len = 10
#     bool_mb_gaze = osm_word_emb
#     bigram_file = 'model/'+model_ID+'-bigram.txt'
#     # hcfeat_file = 'model/'+model_ID+'-hcfeat.txt'
#     START_WORD = 'hhh' #'start_string_taghu' # 'hhh'
#     start_time = time.time()
#     bigram_model = load_bigram_model(bigram_file)
#     # bigram_model = {}
#     if bool_debug:
#         print('load_bigram_model', time.time()-start_time)
#     if bool_mb_gaze:
#         gazetteer_emb_file = 'data/osm_vector'+str(osmembed)+'.txt'
#         gazetteer_emb,gaz_emb_dim = load_embeding(gazetteer_emb_file)
#     else:
#         gazetteer_emb = []
#         gaz_emb_dim = 0
#     # category_words_tuple = [tuple([word]) for word in category_words]
#     # general_words = list(set(general_words).difference(set(category_words_tuple)))
#     # print(general_words)
#     file_name = 'data/osm_abbreviations_globe.csv'
#     abbr_dict = abbrevison1(file_name)
#     start_time = time.time()

#     # char_hc_emb,_ = load_embeding(hcfeat_file)
#     char_hc_emb = {}
#     if bool_debug:
#         print('hcfeat_file', time.time()-start_time)
#     start_time = time.time()

#     word_idx_file = 'model/'+model_ID+'-vocab.txt'
#     word2idx, max_char_len = load_word_index(word_idx_file)
#     if bool_debug:
#         print('load_word_index', time.time()-start_time)
#     start_time = time.time()

#     max_char_len = 20
#     if emb==4:
# #        BertEmbed = BertEmbeds('data/uncased_vocab.txt', 'data/uncased_bert_vectors.txt')
# #        glove_emb, emb_dim = BertEmbed.load_bert_embedding()
#         glove_emb = {}    
#         emb_dim = 1024
#     else:
# #        glove_emb_file = 'data/glove.6B.50d.txt'
# #        glove_emb, emb_dim = load_embeding(glove_emb_file)
#         glove_emb = {}    
#         emb_dim = 50
# #    print('load_embeding', time.time()-start_time)

#     weight_l = emb_dim+gaz_emb_dim+6
#     weights_matrix = np.zeros((len(word2idx.keys()), weight_l))
#     weights_matrix= torch.from_numpy(weights_matrix)
#     tag_to_ix = {"p": 0, "n": 1}
#     HIDDEN_DIM = hidden
#     model_path = 'model/'+model_ID+'epoch'+str(epoch)+'.pkl'
#     DROPOUT = 0.5
#     flex_feat_len = 3
#     fileter_l = filter_l
#     start_time = time.time()

#     model = C_LSTM(weights_matrix, HIDDEN_DIM, fileter_l, lstm_dim, len(tag_to_ix), flex_feat_len, DROPOUT)
#     model.load_state_dict(torch.load(model_path,map_location='cpu'))
#     model.eval()
#     if bool_debug:
#         print('load_state_dict', time.time()-start_time)
#     np_word_embeds = model.embedding.weight.detach().numpy() 
    
    index_t = 0
#    if no_bert:
#        F=1
#    else:
#        F=4
    raw_result_file = 'experiments/result_'+time_str+'m'+gazpne2.model_ID+'region'+str(region)+'th'+str(thres)+'.txt'
    save_file = open(raw_result_file,'w') 
    # save_file.write(model_path)
    save_file.write('\n')
    
    ignored_place_named = {}
    ignored_place_named[6] = [('new','york'),('nyc',),('new','york','city'),('ny')]
    ignored_place_named[7] = [('new','zealand'),('nz',),('uk'),('christchurch',),('chch',),('lyttleton',), ('southland',),('wellington',), ('south','island')]
    if region in ignored_place_named.keys(): 
        ignored_places = ignored_place_named[region]
    else:
        ignored_places = []

    true_count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0
    place_lens = {} 
    detected_score = {}
    hashtag_ignored_list = [21]
    mulit_hashtag_ignore = [18,19,20,21]
    bert_cache = {}
    tweet_cache, valid_keys, temp_tags, total_tweet_count = load_cache_from_file(region,gazpne2.word2idx,gazpne2.abv_punk,input_file,strings)
    tweet_count = 0
    total_return_results = {}
    for key in valid_keys:
        try:
            tweet_count += 1
    #        tweet = js_data[key]['text']
            place_names = tweet_cache[key][0]
            place_offset = tweet_cache[key][1]
            raw_sentences = tweet_cache[key][2]
            offsets = tweet_cache[key][3]
            full_offset= tweet_cache[key][4]
            sentences = tweet_cache[key][5]
            tweet = tweet_cache[key][6]
            hashtag_offsets = tweet_cache[key][7]
            dis_split = tweet_cache[key][8]
            amb_place_names = tweet_cache[key][9]
            amb_place_offset = tweet_cache[key][10]
    
            if region in mulit_hashtag_ignore:
                cur_hash = hashtag_offsets
            else:
                cur_hash = []
    
            truth=[]
            truth.append(str(key))
            for ti, text in enumerate(place_names):
                truth.append(place_offset[ti][0])
                truth.append(place_offset[ti][1])
                pla = ''
                for token in text:
                    pla += token + ' '
                truth.append(pla)
                
            truth_all.append(truth)
            tag_lists = temp_tags[key]
            if not no_bert and region != 100:
                print('#'*50)
                print(str(tweet_count)+'-th tweet' )
                print(tweet)
                #print('ground truth', place_names)
                #print('ground truth', place_offset)
            if bool_debug:
                print(tag_lists)
            new_full_offset = lowerize(offsets, full_offset, tag_lists)
            save_file.write('#'*50)
            save_file.write('\n')
            save_file.write(str(key)+': '+tweet+'\n')
            save_file.write(str(key)+'\n')

            ps = ''
            for place in place_names:
                for w in place:
                    ps += str(w) + ' '
                ps += '\n'
            save_file.write(ps)
    #            pos_str = " ".join(str(item) for item in tag_lists)
    #            save_file.write(pos_str)
    #            save_file.write('\n')
    
    #            last_remove = [] #['area','region']
    #            first_remove = [] #['se','ne','sw','nw']
            
            true_count += len(place_names)
            detected_place_names = []
            detected_offsets = []
            OSM_CONF = postive_pro_t+0.05
            
            # # apply stanford NER
            # mention_index = get_removed_indices_mention(tweet)
            # tweet_nomention = ''.join([tweet[i] for i in range(len(tweet)) if i not in mention_index])
            # words = nltk.word_tokenize(tweet_nomention)
            # str_w = ''
            # for i, word in enumerate(words):
            #         str_w += word + ' '
            # # stanford_offset = align_and_split(tweet, str_w)
            # stanford_result = tagger.tag(words)
            # stanford_result = align_adv(stanford_result, tweet)
            # print(stanford_result)
            stanford_result = []

            # print(tagged)

            stanza_result = stanza_nlp(tweet)
            stanza_ents = stanza_result.entities
            if bool_debug:
                print('stanza',stanza_ents)

            # stanza_ents = []
            invalids = []
            for idx, sentence in enumerate(raw_sentences):
                if sentence:
                    cur_off = offsets[idx]
                    # print(cur_off)
                    
                    sub_index, all_sub_lists, pos_lists = extract_nouns_tweet(tag_lists[idx],gazpne2.s_max_len,dis_split, cur_off)
                    sub_index, all_sub_lists, pos_lists = filterArticles(sentence,sub_index, all_sub_lists, pos_lists)

                    # print(sub_index)
                    # print(all_sub_lists)
                    sub_index, all_sub_lists, pos_lists = filter_invalid_candidates_adv(sub_index, all_sub_lists, \
                                                                                    pos_lists,cur_off, stanza_ents)
                    
                    if not all_sub_lists:
                        continue
                    index_t += 1
    #                    all_sub_lists, sub_index = sub_lists(sentence, s_max_len)
                    osm_probs = [0.1]*len(all_sub_lists)
                    input_emb = np.zeros((len(all_sub_lists),gazpne2.s_max_len,gazpne2.emb_dim+gazpne2.gaz_emb_dim+6*(gazpne2.hc and gazpne2.bool_fix_hc)+gazpne2.flex_feat_len))
                    for i, sub_sen in enumerate(all_sub_lists):
                        sub_sen = [replace_digs(word) for word in sub_sen]
                        sub_sen = [word.lower() for word in sub_sen]
                        input_emb[i] = sentence_embeding(sub_sen, gazpne2.np_word_embeds,gazpne2.word2idx,gazpne2.glove_emb,\
                                                  gazpne2.gazetteer_emb,gazpne2.s_max_len,gazpne2.emb_dim,\
                                                  gazpne2.gaz_emb_dim,gazpne2.max_char_len,gazpne2.bool_mb_gaze,\
                                                 gazpne2.PAD_idx,gazpne2.START_WORD,gazpne2.bigram_model,gazpne2.char_hc_emb,gazpne2.flex_feat_len,gazpne2.hc,gazpne2.bool_fix_hc)
                        if tuple(sub_sen) in gazpne2.osm_names:
                            osm_probs[i] = OSM_CONF
    
                    input_emb= torch.from_numpy(input_emb).float()
                    
                    output = gazpne2.model.predict(input_emb)
                    _, preds_tensor = torch.max(output, 1)
                    pos_prob = torch.sigmoid(output).detach().numpy()
                    pos_prob = pos_prob[:,1]
                    for i, prob in enumerate(pos_prob):
                        if osm_probs[i] > prob:
                            pos_prob[i] = osm_probs[i]
                        if osm_probs[i] == 0:
                            pos_prob[i] = 0
    
                    preds = -preds_tensor.numpy()
                    postives = []
                    general_place_indexs = []
                    for i, p in enumerate(preds):
                         if pos_prob[i] >= postive_pro_t:
                             postives.append(i)
                         else: # some candidates are also checked as long as their pos tag is pronoun
                             all_pron = True
                             for pos in pos_lists[i]:
                                 if pos not in PERSON_POS_ADV and pos not in NUMBER_POS:
                                     all_pron = False
                                     break
                             # if all_pron:
                             if tuple([replace_digs(word).lower() for word in  all_sub_lists[i]]) in general_words and not no_bert:
                                 postives.append(i)
                                 general_place_indexs.append(i)
                    removed_positvie, valid_stanza_loc = sub_stanza(postives,stanza_ents,cur_off,sub_index,all_sub_lists)
                    if bool_debug:
                        for item in removed_positvie:
                            print('stanza_sub',all_sub_lists[item])
                    postives = [item for item in postives if item not in removed_positvie]
                    temp_positive = []
                    
                    for i, p in enumerate(preds):
                         if i not in postives:
                                 all_pron = True
                                 if all_pron or 1:
                                   cur_off_pla = tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]])
                                   # for item in all_sub_lists[i]:
                                   #     if item not in word2idx.keys():
                                   #         break
                                   bool_overlap = False
                                   for j in postives:
                                         cur_off_pla_j = tuple([cur_off[sub_index[j][0]][0],cur_off[sub_index[j][-1]][1]])
                                         if is_overlapping(cur_off_pla[0], cur_off_pla[1], cur_off_pla_j[0], cur_off_pla_j[1]):
                                             bool_overlap = True
                                             break
                                   if bool_overlap:
                                         continue
                                     
                                   if len(all_sub_lists[i]) > 1 and inStanzaLoc(stanza_ents, all_sub_lists[i], cur_off_pla, 1):
                                        if pos_prob[i] >= 0.5:
                                        # if not [item for item in all_sub_lists[i] if item.lower() not in word2idx.keys()]:
                                            temp_positive.append(i)
                                            if bool_debug:
                                              print('stanza_pob',all_sub_lists[i],pos_prob[i])
                                        else:
                                              valid_loc, bert_cache = general_check(bert_cache,gazpne2,cur_off_pla,\
                                                                                    special_ent_t,loc_thres,\
                                                                                    all_sub_lists[i],new_full_offset,\
                                                                                    pos_lists[i],bool_debug,weight)
                                              if valid_loc:
                                                  temp_positive.append(i)
                                                  if bool_debug:
                                                      print('stanza_str1',all_sub_lists[i])
                                              else:
                                                  if bool_debug:
                                                      print('stanza_str_invalids',all_sub_lists[i])
                    postives.extend(temp_positive)
                    origin_pos_prob = pos_prob 
                    pos_prob = pos_prob[postives]
                    sort_index = (-pos_prob).argsort()
                    sort_index = sort_index.tolist()
                    index = 1
                    while len(postives) > index:
                        bool_overlap = 0
                        for s_index in sort_index:
                            if overlap(sub_index[postives[sort_index[index]]], sub_index[postives[s_index]]):
                                bool_overlap = 1
                                break
                        if not bool_overlap:
                            for i in range(index):
                                if is_Sublist(sub_index[postives[sort_index[index]]],sub_index[postives[sort_index[i]]]):
                                    cur_index = sort_index[index]
                                    del sort_index[index]
                                    sort_index.insert(i, cur_index)
                        index += 1
                    
                    detected_index = []
                    final_sub_sen = []
                    for index in sort_index:
                        final_sub_sen.append(postives[index])
    #                    print ('start_time4',len(all_sub_lists), time.time()-start_time4)
                    cur_results = {}
                    for i in final_sub_sen:
                        bool_added = True
                        for p in detected_index:
                            if  (intersection(sub_index[p], sub_index[i]) and \
                                        not (is_Sublist(sub_index[i],sub_index[p]))) or (is_Sublist(sub_index[p],sub_index[i])):
                                bool_added = False
                                break
                        if bool_added:
                            if no_bert:
                                detected_index.append(i)
                                continue
                            
                            # estimate extrinsic probability    
                            cur_off_pla = tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]])
                            # judge if it is also detected as location by stanza.
                            
                            if inStanzaLoc(stanza_ents,all_sub_lists[i],cur_off_pla,1):
                                detected_index.append(i)
                                continue
                            # if inStanzaLoc(stanza_ents,all_sub_lists[i],cur_off_pla,1,\
                            #                non_LOC_NER_TYPEs, 2):
                            #     valid_loc, bert_cache = general_check(bert_cache,obj,cur_off_pla,\
                            #           special_ent_t,loc_thres+0.3,\
                            #           all_sub_lists[i],new_full_offset,\
                            #           pos_lists[i],bool_debug,weight,1)
                            #     if valid_loc:
                            #         detected_index.append(i)
                            #         print('stanza_person_loc',all_sub_lists[i])
                            #     continue
                                    
                            
                            bool_general = False
                            if i in general_place_indexs:
                                bool_general = True
                            if bool_general:
                                continue
                            masked_sentence, cap_sen, ori_masked_sen = gen_mask_sentence3(all_sub_lists[i])
                            masked_context_sentence = gen_mask_sentence2(cur_off_pla, new_full_offset, pos_lists[i])
                            if bool_debug:
                                print(masked_context_sentence)
                            context_ent_prob, context_ent_prob_gen,context_descs = gazpne2.context_cue_new(masked_context_sentence,1,cap_sen,'',bool_debug,bool_formal)
                            context_ent_prob = pure_ent(context_ent_prob)
                            if bool_debug:
                                # print(all_sub_lists[i])
                                print('context_ent_prob', context_ent_prob)
                            if not weight:
                                context_ent_prob = context_ent_prob_gen
                            
                            bool_detected = fusion_strategy29({'LOC':0},context_ent_prob, ent_thres, \
                                                              context_thres, all_sub_lists[i], \
                                                              gazpne2.abbr_dict.keys(), bool_general, \
                                                              pos_lists[i], abb_context_thres, \
                                                              merge_thres,num_context_thres, \
                                                              single_person_c_t)
                           
                            contain_number = 0
                            for item in all_sub_lists[i]:
                                if hasNumbers(item):
                                    contain_number = 1
                                    break
                            ent_prob = {}
                            if (not bool_fast) or( not bool_detected and not contain_number):
                                if bool_debug:
                                    print(masked_sentence)
                                if masked_sentence in bert_cache.keys():
                                    ent_prob = bert_cache[masked_sentence][0]
                                    ent_prob_gen = bert_cache[masked_sentence][1]
                                else:
                                    ent_prob, ent_prob_gen ,descs = gazpne2.context_cue_new(masked_sentence,0,cap_sen, ori_masked_sen,bool_debug,bool_formal)
                                    bert_cache[masked_sentence] =  [ent_prob,ent_prob_gen]
                                    if len(cap_sen)>1:
                                        ent_prob['LOC']+=0.15
                                ent_prob = pure_ent(ent_prob)
                                ent_prob_gen = pure_ent(ent_prob_gen)
                                if bool_debug:
                                    print('ent_prob', ent_prob)
                                bool_detected = fusion_strategy29(ent_prob,context_ent_prob, ent_thres, context_thres, all_sub_lists[i], abbr_dict.keys(), bool_general, pos_lists[i], abb_context_thres, merge_thres,num_context_thres, single_person_c_t)    
                            bool_inter = 0
                            for hashtag in hashtag_offsets:
                                if intersection(list(range(hashtag[0],hashtag[1]+1)), list(range(cur_off_pla[0],cur_off_pla[1]+1))):
                                    bool_inter = 1
                                    break
                            bool_fc, fc_count = FC_check(fc_tokens,context_descs,all_sub_lists[i],fc_ratio)
                            if bool_detected and  not (region in hashtag_ignored_list and bool_inter):
                                detected_index.append(i)
                            if not bool_fast:    
                                result_mid = create_result(key, '0', all_sub_lists[i], pos_lists[i], cur_off_pla,ent_prob,\
                                                           context_ent_prob,{},{},\
                                                           bool_general,0,float(fc_count)/float(len(context_descs)), origin_pos_prob[i],amb_place_offset, amb_place_names, cur_hash)
                                cur_results[i]=result_mid
                          
                    if not no_bert and bool_special_check:
                        #find the sub list that can be used for special check
                        check_candidates = []
                        remove_child = []
                        for i in range(len(all_sub_lists)):
                            bool_all_prop = 1
                            for pos in pos_lists[i]:
                                if not pos in PERSON_POS:
                                    bool_all_prop = 0
                                    break
                            if bool_all_prop and i not in final_sub_sen and i not in detected_index \
                                              and not index_list_int(i,detected_index,sub_index) and len(all_sub_lists[i]) == 1: #and not index_list_int(i,final_sub_sen,sub_index) 
                                                  #and ( len(all_sub_lists[i]) == 1 and  all_sub_lists[i][0].lower()  not in category_words) 
                                             child_index = index_list_sub(i,check_candidates,sub_index)
                                             remove_child.extend(child_index)
                                             check_candidates.append(i)
                        real_candidate = [item for item in check_candidates if item not in remove_child]
                        real_candidate = list(set(real_candidate))
                        for i in real_candidate:
                            #if 1: #i not in final_sub_sen and i not in detected_index and len(all_sub_lists[i])==1 and pos_lists[i][0] in \
                                   #          PERSON_POS_ADV and all_sub_lists[i][0].lower() not in category_words:
                            bool_check = 1
                            for k in final_sub_sen:
                                if sub_index[i][0] in sub_index[k]:
                                    bool_check = 0
                                    break
                            if bool_check:
                                    cur_off_pla = tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]])
                                    if bool_debug:
                                        print('entity', all_sub_lists[i],cur_off_pla)
                                    if inStanzaLoc(stanza_ents,all_sub_lists[i],cur_off_pla,special=1):
                                        add_t = 0
                                        for item in all_sub_lists[i]: 
                                            if item.lower() not in word2idx.keys():
                                                add_t = 0.1
                                                break
                                        valid_loc, bert_cache = general_check(bert_cache,gazpne2,cur_off_pla,\
                                        special_ent_t,loc_thres+add_t,\
                                        all_sub_lists[i],new_full_offset,\
                                        pos_lists[i],bool_debug,weight,1)
                                        if valid_loc:
                                            detected_index.append(i)
                                            if bool_debug:
                                                print('stanza_special1:',all_sub_lists[i])
                                        else:
                                            if bool_debug:
                                                print('stanza_special1_invalids:',all_sub_lists[i])
                                        continue

                                    # continue
                                    # masked_sentence,cap_sen, ori_masked_sen = gen_mask_sentence3(all_sub_lists[i])
                                    # if masked_sentence in bert_cache.keys():
                                    #     ent_prob = bert_cache[masked_sentence][0]
                                    #     ent_prob_gen = bert_cache[masked_sentence][1]
                                    #     ent_prob = pure_ent(ent_prob)
                                    #     if ent_prob['LOC'] < special_ent_t:
                                    #         break
                                    # if len(cap_sen) < 2 and (cap_sen[0] in obj.Word_Entities.keys() or cap_sen[0].lower() in obj.Word_Entities.keys()):
                                    #     ent_prob, ent_prob_gen, descs = obj.context_cue_new(masked_sentence,0,cap_sen,ori_masked_sen,bool_debug,bool_formal)
                                    #     ent_prob = pure_ent(ent_prob)
                                    #     if ent_prob['LOC'] < special_ent_t:
                                    #         break
                                    # masked_context_sentence = gen_mask_sentence2(cur_off_pla, new_full_offset, pos_lists[i])
                                    # context_ent_prob, context_ent_prob_gen,context_descs = obj.context_cue_new(masked_context_sentence,1,cap_sen,'',bool_debug,bool_formal)                                        
                                    # context_ent_prob = pure_ent(context_ent_prob)
                                    # if bool_debug:                                            
                                    #     print('special check:context_ent_prob', context_ent_prob)

                                    # if not weight:
                                    #     context_ent_prob = context_ent_prob_gen
                                    # bool_fc, fc_count = FC_check(fc_tokens,context_descs,all_sub_lists[i],fc_ratio)

                                    # if (not bool_fast) or (context_ent_prob['LOC'] >= loc_thres):
                                    #     masked_sentence,cap_sen, ori_masked_sen = gen_mask_sentence3(all_sub_lists[i])
                                    #     if masked_sentence in bert_cache.keys():
                                    #         ent_prob = bert_cache[masked_sentence][0]
                                    #         ent_prob_gen = bert_cache[masked_sentence][1]
                                    #     else:
                                    #         ent_prob, ent_prob_gen, descs = obj.context_cue_new(masked_sentence,0,cap_sen,ori_masked_sen,bool_debug,bool_formal)
                                    #         bert_cache[masked_sentence] =  [ent_prob,ent_prob_gen]
                                    #     ent_prob = pure_ent(ent_prob)
                                    #     if bool_debug:                                                                                            
                                    #         print('special check:ent_prob', ent_prob)
    
                                    #     if ent_prob['LOC'] >= special_ent_t:                                                
                                    #         bool_inter = 0
                                    #         for hashtag in hashtag_offsets:
                                    #             if intersection(list(range(hashtag[0],hashtag[1]+1)), list(range(cur_off_pla[0],cur_off_pla[1]+1))):
                                    #                 bool_inter = 1
                                    #                 break
    
                                    #         if not (region in hashtag_ignored_list and bool_inter):
                                    #             detected_index.append(i)
                                    #             print('general_special', all_sub_lists[i])
    
                                    #     if not bool_fast:
                                    #         result_mid = create_result(key, '2', all_sub_lists[i], pos_lists[i], cur_off_pla, \
                                    #                   ent_prob,context_ent_prob,{},{},\
                                    #                   0,0,fc_count/float(len(context_descs)),postive_pro_t+0.02,amb_place_offset, amb_place_names, cur_hash)
                                    #         cur_results[i]=result_mid
                                        
                    overlapped = []
                    real_detected_index = []
                    for i in detected_index:
                        bool_sub = False
                        for j in detected_index:
                            if j != i:
                                if (is_Sublist(sub_index[j],sub_index[i])):
                                    bool_sub = True
                                    break
                        replace_list = []
                        if not bool_sub:
                            for j, sub_sen in enumerate(all_sub_lists):
                                bool_intersect = 0
                                for k in detected_index:
                                    if k!=j and k!=i:
                                        if intersection(sub_index[j], sub_index[k]):
                                            bool_intersect = 1
                                            break
                                # if bool_debug:
                                #     print(bool_intersect,all_sub_lists[i], sub_index,i,j,tag_lists[idx])
                                if not bool_intersect and bool_expansion(sub_index,i,j,lastfix_places_words,final_sub_sen,tag_lists[idx],\
                                                  spatial_indicators,prefix_places_words,exp_pos_list):
                                    # print('replace',all_sub_lists[i],all_sub_lists[j])
                                    bool_add_first = 1
                                    for index in replace_list:
                                        if ( not is_Sublist(sub_index[j],sub_index[index])):
                                            bool_add_first = 0
                                            break
                                    if bool_add_first:
                                        replace_list.insert(0,j)
                                    else:
                                        replace_list.append(j)
                            if replace_list:
                                real_detected_index.append(replace_list[0])
                            else:
                                real_detected_index.append(i)
    
    #                                sub_sen = [replace_digs(word) for word in sub_sen]
    #                                sub_sen = [word.lower() for word in sub_sen]
    #                                if cur_off[sub_index[j][-1]][1] == cur_off[sub_index[i][-1]][1] \
    #                                     and cur_off[sub_index[i][0]][0] > cur_off[sub_index[j][0]][0] \
    #                                     and len(sub_sen) -1 == len(all_sub_lists[i]) and sub_sen[0] in prefix_places_words:
    #                                         if not bool_fast:
    #                                             cur_results[i]['s_idx'] = cur_off[sub_index[j][0]][0]
    #                                             cur_results[i]['e_idx'] = cur_off[sub_index[j][-1]][1]
    #                                             name=''
    #                                             for t in all_sub_lists[j]:
    #                                                 name += t + ' '
    #                                             pos_name = ''
    #                                             for t in pos_lists[j]:
    #                                                 pos_name += t + ' '
    #                                             cur_results[i]['text'] = name                                    
    #                                             cur_results[i]['pos'] = pos_name
    #                                         i = j
    #                                         break
    #                                elif cur_off[sub_index[j][0]][0] == cur_off[sub_index[i][0]][0] \
    #                                     and cur_off[sub_index[j][-1]][1] > cur_off[sub_index[i][-1]][1] \
    #                                     and len(sub_sen) -1 == len(all_sub_lists[i]) and sub_sen[-1] in lastfix_places_words:
    #                                         if not bool_fast:
    #                                             cur_results[i]['s_idx'] = cur_off[sub_index[j][0]][0]
    #                                             cur_results[i]['e_idx'] = cur_off[sub_index[j][-1]][1]
    #                                             name=''
    #                                             for t in all_sub_lists[j]:
    #                                                 name += t + ' '
    #                                             pos_name = ''
    #                                             for t in pos_lists[j]:
    #                                                 pos_name += t + ' '
    #                                             cur_results[i]['text'] = name                                    
    #                                             cur_results[i]['pos'] = pos_name
    #                                         i = j
    #                                         break
                    
                    for i in real_detected_index:
                        bool_sub = False
                        for j in real_detected_index:
                            if j != i:
                                if (is_Sublist(sub_index[j],sub_index[i])):
                                    bool_sub = True
                                    break
                        if not bool_sub:
                            detected_place_names.append(tuple(all_sub_lists[i]))
                            detected_offsets.append(tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]]))
                            # save_file.write(str(all_sub_lists[i])+'\n')
                            # save_file.write(str(all_sub_lists[i])+'\n')
                            # save_file.write(str(round(origin_pos_prob[i],3))+':'+str(all_sub_lists[i])+'\n')
                     
                    #invalids.extend(include_valid_places(real_detected_index, sub_index, cur_off, stanza_ents))

                    # for i in detected_index:
                    #     bool_sub = False
                    #     for j in detected_index:
                    #         if j != i:
                    #             if (is_Sublist(sub_index[j],sub_index[i])):
                    #                 bool_sub = True
                    #                 break
                    #     if not bool_sub:
                    #         for j, sub_sen in enumerate(all_sub_lists):
                    #             sub_sen = [replace_digs(word) for word in sub_sen]
                    #             sub_sen = [word.lower() for word in sub_sen]
                    #             if cur_off[sub_index[j][-1]][1] == cur_off[sub_index[i][-1]][1] \
                    #                  and cur_off[sub_index[i][0]][0] > cur_off[sub_index[j][0]][0] \
                    #                  and len(sub_sen) -1 == len(all_sub_lists[i]) and sub_sen[0] in prefix_places_words:
                    #                      if not bool_fast:
                    #                          cur_results[i]['s_idx'] = cur_off[sub_index[j][0]][0]
                    #                          cur_results[i]['e_idx'] = cur_off[sub_index[j][-1]][1]
                    #                          name=''
                    #                          for t in all_sub_lists[j]:
                    #                              name += t + ' '
                    #                          pos_name = ''
                    #                          for t in pos_lists[j]:
                    #                              pos_name += t + ' '
                    #                          cur_results[i]['text'] = name                                    
                    #                          cur_results[i]['pos'] = pos_name
                    #                      i = j
                    #                      break
                    #             elif cur_off[sub_index[j][0]][0] == cur_off[sub_index[i][0]][0] \
                    #                  and cur_off[sub_index[j][-1]][1] > cur_off[sub_index[i][-1]][1] \
                    #                  and len(sub_sen) -1 == len(all_sub_lists[i]) and sub_sen[-1] in lastfix_places_words:
                    #                      if not bool_fast:
                    #                          cur_results[i]['s_idx'] = cur_off[sub_index[j][0]][0]
                    #                          cur_results[i]['e_idx'] = cur_off[sub_index[j][-1]][1]
                    #                          name=''
                    #                          for t in all_sub_lists[j]:
                    #                              name += t + ' '
                    #                          pos_name = ''
                    #                          for t in pos_lists[j]:
                    #                              pos_name += t + ' '
                    #                          cur_results[i]['text'] = name                                    
                    #                          cur_results[i]['pos'] = pos_name
                    #                      i = j
                    #                      break
    
                    #         detected_place_names.append(tuple(all_sub_lists[i]))
                    #         detected_offsets.append(tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]]))
                    #         save_file.write(str(round(origin_pos_prob[i],3))+':'+str(all_sub_lists[i])+'\n')
    #                    print ('start_time1',time.time()-start_time1)
                    if not bool_fast:
                        for index_s in cur_results:
                             result_mids.append(cur_results[index_s])
            #stanza_offs, satnza_places = valid_stanza_places(invalids, stanza_ents)
            #if bool_debug:
            #    print('stanza',satnza_places)
            #for off, place in zip(stanza_offs, satnza_places):
            #    detected_place_names.append(place)
            #    detected_offsets.append(off)
            #    save_file.write('stanza:'+str(place)+'\n')
            # print(stanford_offset)
            # print(detected_offsets)
            
            new_offsets, new_places = valid_stanza_places(stanza_ents, detected_place_names, detected_offsets, word2idx.keys()) #word2idx.keys()
            
            # # new_offsets, mew_places = valid_stanford_places(stanford_result, detected_place_names, detected_offsets)                
            # if bool_debug:
            #     print('stanford/stanza',new_places)
            for off, place in zip(new_offsets, new_places):
                valid_loc, bert_cache = general_check(bert_cache,gazpne2,off,\
                                            special_ent_t,loc_thres+0.1,\
                                            place,new_full_offset,\
                                            [],bool_debug,weight)
                if valid_loc:
                    if bool_debug:
                        print('stanza_full',place)
                    detected_place_names.append(tuple(place))
                    detected_offsets.append(off)
                    # save_file.write('stanford/stanza:'+str(place)+'\n')

            c_tp, c_fp, c_fn, place_detect_score = interset_num(detected_offsets,place_offset,detected_place_names,\
                                                               place_names,ignored_places, amb_place_offset, amb_place_names, cur_hash)
            detection_dicts = []
            for i, item in enumerate(detected_place_names):
                detection_dict = {}
                place_str = ''
                for j, w in enumerate(item):
                    place_str += w
                    if j != len(item)-1:
                        place_str += ' '
                detection_dict['LOC'] = place_str
                detection_dict['offset'] = detected_offsets[i]
                detection_dicts.append(detection_dict)
            detection_dicts = sorted(detection_dicts, key = lambda i: i['offset'][0])
            total_return_results[key]=detection_dicts
                
                # save_file.write('tp:'+str(c_tp)+' c_fp:'+str(c_fp)+' c_fn:'+str(c_fn))
            save_file.write('\n')
            if not no_bert and region != 100:
                print(detected_place_names)
                print(detected_offsets)
    #                print(place_offset)
    
            for p, i in enumerate(place_names):
                cur_len_p = 0
                for pp in i:
                    if hasNumbers(pp):
                        groups = re.split('(\d+)',pp)
                        groups = [x for x in groups if x]
                        cur_len_p += len(groups)
                    else:
                        segments = segment(pp)
                        cur_len_p += len(segments)
                if cur_len_p in place_lens.keys():
                    place_lens[cur_len_p] += 1
                    detected_score[cur_len_p] += place_detect_score[p]
                else:
                    place_lens[cur_len_p] = 1
                    detected_score[cur_len_p] = place_detect_score[p]
            TP_count += c_tp
            FP_count += c_fp
            FN_count += c_fn
        except BaseException as e:
            total_return_results[key]=[]
            print(e)
            print('exception occurs')
    if TP_count+FP_count > 0 and TP_count+FN_count > 0:
        P = TP_count/(TP_count+FP_count) 
        R = TP_count/(TP_count+FN_count) 
        F = (2*P*R) / (P+R)
        if region != 100:
            print(TP_count,FP_count,FN_count)
            print('P',P,'R',R,'F',F)
            print('true count:', true_count)
            print('tweet count', total_tweet_count)
        save_file.write('recall:' + str(R))
        save_file.write('\n')
        save_file.write('precision:' + str(P))
        save_file.write('\n')
        save_file.write('f1:' +  str(F))
        save_file.write('\n')
        save_file.write('TP:' + str(TP_count))
        save_file.write('\n')
        save_file.write('FP:' + str(FP_count))
        save_file.write('\n')
        save_file.write('FN:' + str(FN_count))
        save_file.write('\n')
        save_file.write('true count:' + str(true_count))
        save_file.write('\n')
        save_file.write('tweet count:' + str(total_tweet_count))
        save_file.write('\n')

        save_file.write(json.dumps(detected_score)) # use `json.loads` to do the reverse
        save_file.write(json.dumps(place_lens)) # use `json.loads` to do the reverse
        detection_rate = [detected_score[key]/place_lens[key] for key in place_lens.keys()]
        for item in detection_rate:
            save_file.write("%s\n" % item)
    else:
        if region != 100:
            print(place_lens)
            print(TP_count,FP_count,FN_count)
        P = 0
        F = 0
        R = 0
    save_file.close()


    if not bool_fast:
        """save into csv"""
        csv_file = 'data/'+model_ID+str(epoch)+time_str+'region'+str(region)+'.csv'
        fieldnames = []
        for item in result_mids:
            for key in item.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        if os.path.exists(csv_file):
            os.remove(csv_file);
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for dic in result_mids:
                    writer.writerow(dic)
    
        """save into csv"""
         
        csv_file = 'data/'+'region'+str(region)+'.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file);
        with open(csv_file, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for dic in truth_all:
                wr.writerow(dic)
    return F,P,R,total_return_results

